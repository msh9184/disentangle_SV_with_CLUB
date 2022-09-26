#!/usr/bin/python
#-*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, pdb, sys, random, time, os, itertools, shutil, importlib
import numpy as np
from tuneThreshold import tuneThresholdfromScore
from DatasetLoader import test_dataset_loader, infer_dataset_loader
from torch.cuda.amp import autocast, GradScaler
from torch.optim.swa_utils import SWALR, AveragedModel

class WrappedModel(nn.Module):
    def __init__(self, model):
        super(WrappedModel, self).__init__()
        self.module = model

    def forward(self, x, label=None):
        return self.module(x, label)


class SpeakerNet(nn.Module):
    def __init__(self, model, optimizer, trainfunc, num_out, num_class_spk, num_class_dev, margin, scale, num_utt, **kwargs):
        super(SpeakerNet, self).__init__()
        # networks
        SpeakerNetModel = importlib.import_module('models.'+model).__getattribute__('MainModel')
        self.__S__ = SpeakerNetModel(**kwargs)

        FactorizationModel = importlib.import_module('models.Factorization2').__getattribute__('MainModel')
        self.__F__ = FactorizationModel(num_out=num_out)

        # classifier: objective functions
        LossFunction = importlib.import_module('loss.'+trainfunc).__getattribute__('LossFunction')
        self.__L_cls_spk__ = LossFunction(num_out=num_out, num_class=num_class_spk, margin=margin, scale=scale)

        LossFunction = importlib.import_module('loss.aamsoftmax').__getattribute__('LossFunction')
        self.__L_cls_dev__ = LossFunction(num_out=num_out, num_class=num_class_dev)

        # MI estimator: objective functions
        MIEstimator = importlib.import_module('mi_estimators.CLUBForCategorical').__getattribute__('MIEstimator')
        self.__MI_ctg_dev__ = MIEstimator(input_dim=num_out, label_num=num_class_dev)

        MIEstimator = importlib.import_module('mi_estimators.CLUBForCategorical').__getattribute__('MIEstimator')
        self.__MI_ctg_spk__ = MIEstimator(input_dim=num_out, label_num=num_class_spk)

        MIEstimator = importlib.import_module('mi_estimators.CLUB').__getattribute__('MIEstimator')
        self.__MI_spk2dev__ = MIEstimator(x_dim=num_out, y_dim=num_out, hidden_size=1024)

        self.num_utt = num_utt


    def forward(self, data, label_spk=None, label_dev=None, opt=None, is_infer=False):

        if label_spk == None:
            x = self.__S__.forward(data.reshape(-1,data.size()[-1]).cuda(), aug=False)
            x_spk, x_dev = self.__F__.forward(x)

            # inference
            if is_infer:
                return x_spk, x_dev
            # evaluation
            else:
                return x_spk

        else:
            data = data.reshape(-1, data.size()[-1]).cuda()
            if opt=='optimizing_estimators':
                with torch.no_grad():
                    x = self.__S__.forward(data, aug=True)
                    x_spk, x_dev = self.__F__.forward(x)

                nnll_ctg_dev = self.__MI_ctg_dev__.learning_loss(x_spk, label_dev.repeat_interleave(2))
                nnll_ctg_spk = self.__MI_ctg_spk__.learning_loss(x_dev, label_spk.repeat_interleave(2))
                nnll_spk2dev = self.__MI_spk2dev__.learning_loss(x_spk, x_dev)

                nnll_tot = nnll_ctg_dev + nnll_ctg_spk + nnll_spk2dev

                return nnll_tot, nnll_ctg_dev, nnll_ctg_spk, nnll_spk2dev

            elif opt=='optimizing_networks':
                x = self.__S__.forward(data, aug=True)
                x_spk, x_dev = self.__F__.forward(x)
                nloss_spk, prec_spk = self.__L_cls_spk__.forward(x_spk.reshape(self.num_utt, -1, x_spk.size()[-1]).transpose(1,0).squeeze(1), label_spk)
                nloss_dev, prec_dev = self.__L_cls_dev__.forward(x_dev, label_dev.repeat_interleave(2))

                #with torch.no_grad():
                nmi_ctg_dev = self.__MI_ctg_dev__.forward(x_spk, label_dev.repeat_interleave(2))
                nmi_ctg_spk = self.__MI_ctg_spk__.forward(x_dev, label_spk.repeat_interleave(2))
                nmi_spk2dev = self.__MI_spk2dev__.forward(x_spk, x_dev)

                nloss_tot = 5.0 * nloss_spk + 10.0 * nloss_dev + 0.5 * nmi_ctg_dev + 0.1 * nmi_ctg_spk + 0.1 * nmi_spk2dev
                return nloss_tot, nloss_spk, nloss_dev, nmi_ctg_dev, nmi_ctg_spk, nmi_spk2dev, prec_spk, prec_dev


class ModelTrainer(object):
    def __init__(self, speaker_model, optimizer, scheduler, gpu, **kwargs):
        self.__model__  = speaker_model

        Optimizer = importlib.import_module('optimizer.'+optimizer).__getattribute__('Optimizer')
        self.__optimizer_main__ = Optimizer(list(self.__model__.module.__S__.parameters()) + list(self.__model__.module.__F__.parameters()) + list(self.__model__.module.__L_cls_spk__.parameters()) + list(self.__model__.module.__L_cls_dev__.parameters()), **kwargs)

        Optimizer = importlib.import_module('optimizer.'+optimizer).__getattribute__('Optimizer')
        self.__optimizer_mi__ = Optimizer(list(self.__model__.module.__MI_ctg_dev__.parameters()) + list(self.__model__.module.__MI_ctg_spk__.parameters()) + list(self.__model__.module.__MI_spk2dev__.parameters()), **kwargs)

        Scheduler = importlib.import_module('scheduler.'+scheduler).__getattribute__('Scheduler')
        self.__scheduler_main__, _ = Scheduler(self.__optimizer_main__, **kwargs)

        Scheduler = importlib.import_module('scheduler.'+scheduler).__getattribute__('Scheduler')
        self.__scheduler_mi__, _ = Scheduler(self.__optimizer_mi__, **kwargs)

        self.scaler = GradScaler() 
        self.gpu = gpu
        self.ngpu = int(torch.cuda.device_count())
        self.ndistfactor = int(kwargs.pop('num_utt') * self.ngpu)
        self.num_mi_update = kwargs.pop('num_mi_update')

        self.logfile = open(kwargs.pop('result_save_path')+"/logs_"+ str(gpu) +".txt", "a+")


    def train_network(self, loader, epoch, verbose):
        self.__model__.train()
        #self.__scheduler__.step(epoch-1)
        if epoch==1: self.__scheduler_main__.step(0)
        bs = loader.batch_size
        df = self.ndistfactor
        cnt, idx, loss_tot, loss_spk, loss_dev, acc_spk, acc_dev = 0, 0, 0, 0, 0, 0, 0
        mi_ctg_spk, mi_ctg_dev, mi_spk2dev, nll_ctg_spk, nll_ctg_dev, nll_spk2dev = 0, 0, 0, 0, 0, 0
        tstart = time.time()
        for data, data_spk, data_dev in loader:
            #print(loader.__len__())
            #exit()
            self.__model__.zero_grad()

            data = data.transpose(1,0) 
            label_spk = torch.LongTensor(data_spk).cuda()
            label_dev = torch.LongTensor(data_dev).cuda()

            # MI estimators q(yd|xs), q(ys|xd), q(xd|xs) updates (main networks are fixed)
            opt = 'optimizing_estimators'
            for i in range(self.num_mi_update):
                with autocast():
                    nnll_tot, nnll_ctg_dev, nnll_ctg_spk, nnll_spk2dev = self.__model__(data, label_spk, label_dev, opt=opt)
                self.scaler.scale(nnll_tot).backward()
                self.scaler.step(self.__optimizer_mi__)
                self.scaler.update()
                self.__scheduler_mi__.step()

            nll_ctg_spk += nnll_ctg_spk.detach().cpu().item()
            nll_ctg_dev += nnll_ctg_dev.detach().cpu().item()
            nll_spk2dev += nnll_spk2dev.detach().cpu().item()

            # main networks updates (using classifiers and MI estimators)
            opt = 'optimizing_networks'
            with autocast():
                nloss_tot, nloss_spk, nloss_dev, nmi_ctg_dev, nmi_ctg_spk, nmi_spk2dev, prec_spk, prec_dev = self.__model__(data, label_spk, label_dev, opt=opt)
            self.scaler.scale(nloss_tot).backward()
            self.scaler.step(self.__optimizer_main__)
            self.scaler.update()

            loss_tot += nloss_tot.detach().cpu().item()
            loss_spk += nloss_spk.detach().cpu().item()
            loss_dev += nloss_dev.detach().cpu().item()

            mi_ctg_spk += nmi_ctg_spk.detach().cpu().item()
            mi_ctg_dev += nmi_ctg_dev.detach().cpu().item()
            mi_spk2dev += nmi_spk2dev.detach().cpu().item()

            acc_spk += prec_spk.detach().cpu().item()
            acc_dev += prec_dev.detach().cpu().item()

            cnt += 1
            idx += bs

            lr = self.__optimizer_main__.param_groups[0]['lr']
            self.__scheduler_main__.step() # schd: iteration 2022.04.21.
            telapsed = time.time() - tstart
            tstart = time.time()
            if verbose:
                sys.stdout.write("\rProcessing {:d} of {:d}: Loss_tot {:f}, Loss_cls_spk {:f}, Loss_cls_dev {:f}, Acc_spk {:2.3f}%, Acc_dev {:2.3f}%, MI_ctg_spk {:f}, MI_ctg_dev {:f}, MI_spk2dev {:f}, NLL_ctg_spk {:f}, NLL_ctg_dev {:f}, NLL_spk2dev {:f}, lr {:.8f} - {:.2f} Hz ".format(idx*df, loader.__len__()*bs*df, loss_tot/cnt, loss_spk/cnt, loss_dev/cnt, acc_spk/cnt, acc_dev/cnt, mi_ctg_spk/cnt, mi_ctg_dev/cnt, mi_spk2dev/cnt, nll_ctg_spk/cnt, nll_ctg_dev/cnt, nll_spk2dev/cnt, lr, bs*df/telapsed))
                sys.stdout.flush()
            self.logfile.write("Processing {:d} of {:d}: Loss_tot {:f}, Loss_cls_spk {:f}, Loss_cls_dev {:f}, Acc_spk {:2.3f}%, Acc_dev {:2.3f}%, MI_ctg_spk {:f}, MI_ctg_dev {:f}, MI_spk2dev {:f}, NLL_ctg_spk {:f}, NLL_ctg_dev {:f}, NLL_spk2dev {:f}, lr {:.8f}\n".format(idx*df, loader.__len__()*bs*df, loss_tot/cnt, loss_spk/cnt, loss_dev/cnt, acc_spk/cnt, acc_dev/cnt, mi_ctg_spk/cnt, mi_ctg_dev/cnt, mi_spk2dev/cnt, nll_ctg_spk/cnt, nll_ctg_dev/cnt, nll_spk2dev/cnt, lr))
            self.logfile.flush()
            #if epoch != 1 and cnt == loader.__len__()//10*5:
            #    return (loss_tot/cnt, loss_spk/cnt, loss_dev/cnt, acc_spk/cnt, acc_dev/cnt, lr)
        return (loss_tot/cnt, loss_spk/cnt, loss_dev/cnt, acc_spk/cnt, acc_dev/cnt, lr)


    def evaluateFromList_with_snorm(self, epoch, test_list, test_path, train_list, train_path, score_norm, tta, num_thread, distributed, top_coh_size, eval_frames=0, num_eval=1, **kwargs):
        if distributed:
            rank = torch.distributed.get_rank()
        else:
            rank = 0
        self.__model__.eval()

        ## Eval loader ##
        feats_eval = {}
        tstart = time.time()
        with open(test_list) as f:
            lines_eval = f.readlines()
        files = list(itertools.chain(*[x.strip().split()[-2:] for x in lines_eval]))
        setfiles = list(set(files))
        setfiles.sort()
        test_dataset = test_dataset_loader(setfiles, test_path, eval_frames=eval_frames, num_eval=num_eval, **kwargs)
        if distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
        else:
            sampler = None
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_thread, drop_last=False, sampler=sampler)
        ds = test_loader.__len__()
        gs = self.ngpu
        for idx, data in enumerate(test_loader):
            inp1 = data[0][0].cuda()
            with torch.no_grad():
                ref_feat = self.__model__(inp1).detach().cpu()
            feats_eval[data[1][0]] = ref_feat
            telapsed = time.time() - tstart
            if rank == 0:
                sys.stdout.write("\r Reading {:d} of {:d}: {:.2f} Hz, embedding size {:d}".format(idx*gs, ds*gs, idx*gs/telapsed,ref_feat.size()[1]))
                sys.stdout.flush()

        ## Cohort loader if using score normalization ##
        if score_norm:
            feats_coh = {}
            tstart = time.time()
            with open(train_list) as f:
                lines_coh = f.readlines()
            setfiles = list(set([x.split()[0] for x in lines_coh]))
            setfiles.sort()
            cohort_dataset = test_dataset_loader(setfiles, train_path, eval_frames=0, num_eval=1, **kwargs)
            if distributed:
                sampler = torch.utils.data.distributed.DistributedSampler(cohort_dataset, shuffle=False)
            else:
                sampler = None
            cohort_loader = torch.utils.data.DataLoader(cohort_dataset, batch_size=1, shuffle=False, num_workers=num_thread, drop_last=False, sampler=sampler)
            ds = cohort_loader.__len__()
            for idx, data in enumerate(cohort_loader):
                inp1 = data[0][0].cuda()
                with torch.no_grad():
                    ref_feat = self.__model__(inp1).detach().cpu()
                feats_coh[data[1][0]] = ref_feat
                telapsed = time.time() - tstart
                if rank == 0:
                    if idx==0: print('')
                    sys.stdout.write("\r Reading {:d} of {:d}: {:.2f} Hz, embedding size {:d}".format(idx*gs, ds*gs, idx*gs/telapsed,ref_feat.size()[1]))
                    sys.stdout.flush()
            coh_feat = torch.stack(list(feats_coh.values())).squeeze(1).cuda()
            if self.__model__.module.__L_cls_spk__.test_normalize:
                coh_feat = F.normalize(coh_feat, p=2, dim=1)

        ## Compute verification scores ##
        all_scores, all_labels = [], []
        if distributed:
            ## Gather features from all GPUs
            feats_eval_all = [None for _ in range(0,torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(feats_eval_all, feats_eval)
            if score_norm:
                feats_coh_all = [None for _ in range(0,torch.distributed.get_world_size())]
                torch.distributed.all_gather_object(feats_coh_all, feats_coh)
        if rank == 0:
            tstart = time.time()
            print('')
            ## Combine gathered features
            if distributed:
                feats_eval = feats_eval_all[0]
                for feats_batch in feats_eval_all[1:]:
                    feats_eval.update(feats_batch)
                if score_norm:
                    feats_coh = feats_coh_all[0]
                    for feats_batch in feats_coh_all[1:]:
                        feats_coh.update(feats_batch)

            ## Read files and compute all scores
            for idx, line in enumerate(lines_eval):
                data = line.split()
                enr_feat = feats_eval[data[1]].cuda()
                tst_feat = feats_eval[data[2]].cuda()
                if self.__model__.module.__L_cls_spk__.test_normalize:
                    enr_feat = F.normalize(enr_feat, p=2, dim=1)
                    tst_feat = F.normalize(tst_feat, p=2, dim=1)

                if tta==True and score_norm==True:
                    print('Not considered condition')
                    exit()
                if tta == False:
                    score = F.cosine_similarity(enr_feat, tst_feat)

                if score_norm:
                    score_e_c = F.cosine_similarity(enr_feat, coh_feat)
                    score_c_t = F.cosine_similarity(coh_feat, tst_feat)

                    if top_coh_size == 0: top_coh_size = len(coh_feat)
                    score_e_c = torch.topk(score_e_c, k=top_coh_size, dim=0)[0]
                    score_c_t = torch.topk(score_c_t, k=top_coh_size, dim=0)[0]
                    score_e = (score - torch.mean(score_e_c, dim=0)) / torch.std(score_e_c, dim=0)
                    score_t = (score - torch.mean(score_c_t, dim=0)) / torch.std(score_c_t, dim=0)
                    score = 0.5 * (score_e + score_t)

                elif tta:
                    score = torch.mean(F.cosine_similarity(enr_feat.unsqueeze(-1), tst_feat.unsqueeze(-1).transpose(0,2)))

                all_scores.append(score.detach().cpu().numpy())
                all_labels.append(int(data[0]))
                telapsed = time.time() - tstart
                sys.stdout.write("\r Computing {:d} of {:d}: {:.2f} Hz".format(idx, len(lines_eval), idx/telapsed))
                sys.stdout.flush()
        return (all_scores, all_labels)


    def inference(self, infer_list, infer_path, num_thread, distributed, eval_frames=0, num_eval=1, save_name='save_var/model00000001', **kwargs):
        self.__model__.eval()

        ## Eval loader ##
        tstart = time.time()
        with open(infer_list) as f:
            lines_infer = f.readlines()
        setfiles = [x.split()[0] for x in lines_infer]
        infer_dataset = infer_dataset_loader(setfiles, infer_path, eval_frames=0, num_eval=1, **kwargs)
        infer_loader = torch.utils.data.DataLoader(infer_dataset, batch_size=1, shuffle=False, num_workers=num_thread, drop_last=False, sampler=None)
        ds = infer_loader.__len__()

        embeds_spk = []
        embeds_dev = []
        labels_spk = []
        for idx, data in enumerate(infer_loader):
            audio = data[0][0].cuda()
            with torch.no_grad():
                embed_spk, embed_dev = self.__model__(data=audio[0], is_infer=True)#.detach().cpu().numpy()

            embed_spk = F.normalize(embed_spk, p=2, dim=1)
            embed_dev = F.normalize(embed_dev, p=2, dim=1)

            embeds_spk += [embed_spk.detach().cpu().numpy()]
            embeds_dev += [embed_dev.detach().cpu().numpy()]
            labels_spk += [data[1][0]]

            telapsed = time.time() - tstart
            sys.stdout.write("\r Reading {:d} of {:d}: {:.2f} Hz".format(idx, ds, idx/telapsed))
            sys.stdout.flush()

        tstart = time.time()
        print('')
        np.save(save_name+'_spk.npy', np.array(embeds_spk))
        np.save(save_name+'_dev.npy', np.array(embeds_dev))
        np.save(save_name+'_lb.npy', labels_spk)
        return 0


    def saveParameters(self, path):
        torch.save(self.__model__.module.state_dict(), path)


    def loadParameters(self, path):
        self_state = self.__model__.module.state_dict()
        loaded_state = torch.load(path, map_location="cuda:%d"%self.gpu)
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")
                if name not in self_state:
                    print("{} is not in the model.".format(origname))
                    continue
            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: {}, model: {}, loaded: {}".format(origname, self_state[name].size(), loaded_state[origname].size()))
                continue
            self_state[name].copy_(param)
