from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from phasei_ml import train as train_module
from phasei_ml.evaluate import evaluate
from phasei_ml.fast_loader import VectorizedEventDataset
from torch.utils.data import ConcatDataset


def _tiny_dataset(root: Path, events=20):
    root.mkdir(parents=True, exist_ok=True)
    particles=[]
    for particle, pid in (("gamma", 0), ("e+", 1)):
        d=root/particle; d.mkdir()
        np.save(d/"evt_id.npy",np.arange(events,dtype=np.int64))
        np.save(d/"pid.npy",np.full(events,pid,dtype=np.int8))
        np.save(d/"pdgid.npy",np.full(events,22 if pid==0 else -11,dtype=np.int32))
        np.save(d/"t0.npy",np.zeros(events,dtype=np.float32))
        np.save(d/"Ek0.npy",np.linspace(1,20,events,dtype=np.float32)+(pid*0.3))
        np.save(d/"x0.npy",np.column_stack([np.linspace(-2,2,events),np.linspace(1,3,events),np.linspace(-1,1,events)]).astype(np.float32))
        theta=np.linspace(.1,3.,events); phi=np.linspace(0,2*np.pi,events,endpoint=False)
        p0=np.column_stack([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)]).astype(np.float32)
        np.save(d/"p0.npy",p0)
        masks=np.arange(events,dtype=np.uint8)%8
        np.save(d/"response_mask.npy",masks)
        split=np.full(events,0,dtype=np.uint8); split[-4:-2]=1; split[-2:]=2
        np.save(d/"split.npy",split)
        norm={}
        for bit,det in enumerate(("ttc","scifi","ecal")):
            channel=[]; times=[]; edeps=[]; offsets=[0]
            for mask in masks:
                if mask & (1<<bit):
                    channel.append(int(mask%2)); times.append(float(mask)); edeps.append(.1+float(mask))
                offsets.append(len(channel))
            np.save(d/f"{det}_channel.npy",np.asarray(channel,dtype=np.int32))
            np.save(d/f"{det}_t.npy",np.asarray(times,dtype=np.float32))
            np.save(d/f"{det}_Edep.npy",np.asarray(edeps,dtype=np.float32))
            np.save(d/f"{det}_offsets.npy",np.asarray(offsets,dtype=np.int64))
            norm[{"ttc":"TTC","scifi":"SciFi","ecal":"ECAL"}[det]]={"channel_count":2,"t_mean":0.,"t_std":1.,"log1p_edep_mean":0.,"log1p_edep_std":1.}
        (d/"normalization.json").write_text(json.dumps(norm))
        particles.append({"particle":particle,"normalization":norm})
    (root/"manifest.json").write_text(json.dumps({"schema_version":1,"particles":particles}))


def test_target_scaler_uses_only_selected_training_rows(tmp_path):
    _tiny_dataset(tmp_path)
    indices={p:np.arange(16) for p in ("gamma","e+")}
    scaler=train_module.fit_target_scaler(tmp_path,indices)
    expected=np.concatenate([np.load(tmp_path/p/"Ek0.npy")[:16] for p in ("gamma","e+")])
    assert scaler["energy"]["mean"]==pytest.approx(float(expected.mean()))
    assert scaler["energy"]["std"]==pytest.approx(float(expected.std()))
    x=np.concatenate([np.load(tmp_path/p/"x0.npy")[:16] for p in ("gamma","e+")])
    np.testing.assert_allclose(scaler["position"]["mean"],x.mean(0),rtol=1e-6,atol=1e-7)
    np.testing.assert_allclose(scaler["position"]["std"],x.std(0),rtol=1e-6,atol=1e-7)
    # Last four rows contain deliberately different targets and must not affect scaling.
    assert scaler["energy"]["mean"] != pytest.approx(float(np.concatenate([np.load(tmp_path/p/"Ek0.npy") for p in ("gamma","e+")]).mean()))


def test_collate_standardizes_targets_and_round_trips(tmp_path):
    _tiny_dataset(tmp_path)
    scaler={"energy":{"mean":10.,"std":2.},"position":{"mean":[1.,2.,3.],"std":[2.,4.,5.]}}
    item={"pid":0,"Ek0":14.,"x0":np.asarray([3.,6.,8.],dtype=np.float32),"p0":np.asarray([0.,0.,1.]),"direction":np.asarray([0.,0.,1.]),"response_mask":0}
    for det in ("ttc","scifi","ecal"):
        item[det]={"channel":np.asarray([],dtype=np.int64),"t":np.asarray([],dtype=np.float32),"log_edep":np.asarray([],dtype=np.float32),"weight":np.asarray([],dtype=np.float32),"mask":np.asarray([],dtype=np.float32)}
    _,target=train_module._collate([item],scaler)
    assert target["energy"].item()==pytest.approx(2.)
    torch.testing.assert_close(target["position"],torch.tensor([[1.,1.,1.]]))


def test_vectorized_batches_match_legacy_for_mixed_and_zero_hit_events(tmp_path):
    _tiny_dataset(tmp_path)
    selected={"gamma":np.arange(16,dtype=np.int64),"e+":np.arange(16,dtype=np.int64)}
    scaler=train_module.fit_target_scaler(tmp_path,selected)
    fast=VectorizedEventDataset(tmp_path,selected,scaler)
    legacy=ConcatDataset([train_module.EventData(tmp_path/name,idx,scaler)
                          for name,idx in selected.items()])
    # Include shuffled and repeated indexes as well as zero-hit response masks.
    indices=[16,0,23,7,7,31,8,15]
    fast_batch,fast_target=fast.__getitems__(indices)
    legacy_batch,legacy_target=train_module._collate([legacy[i] for i in indices],scaler)
    for name in ("response_mask","ttc","scifi","ecal"):
        if isinstance(fast_batch[name],dict):
            for field in fast_batch[name]:
                torch.testing.assert_close(fast_batch[name][field],legacy_batch[name][field])
        else:
            torch.testing.assert_close(fast_batch[name],legacy_batch[name])
    for field in fast_target:
        torch.testing.assert_close(fast_target[field],legacy_target[field])
    assert not fast_batch["ttc"]["mask"][0].any()
    assert not fast_batch["scifi"]["mask"][0].any()
    assert not fast_batch["ecal"]["mask"][0].any()


def test_train_never_selects_test_split_and_resume_reads_last_checkpoint(tmp_path,monkeypatch):
    data=tmp_path/"data"; _tiny_dataset(data)
    original=train_module._select_indices
    calls=[]
    def guarded(directory,split_id,limit,seed):
        calls.append(split_id)
        assert split_id != 2
        return original(directory,split_id,limit,seed)
    monkeypatch.setattr(train_module,"_select_indices",guarded)
    out=tmp_path/"runs"/"best.pt"; last=tmp_path/"runs"/"last.pt"
    result=train_module.train(data,out,epochs=1,batch_size=16,patience=5,device="cpu",last_output=last)
    assert set(calls)=={0,1}
    assert out.exists() and last.exists()
    best=torch.load(out,map_location="cpu",weights_only=False)
    assert "target_scaler" in best and "validation_by_head" in best["history"][0]
    assert "test_metrics" not in best
    resumed=train_module.train(data,out,epochs=2,batch_size=16,patience=5,device="cpu",last_output=last,resume=last)
    last_state=torch.load(last,map_location="cpu",weights_only=False)
    assert last_state["epoch"]==2
    assert resumed["epochs_completed"]==2
    model=train_module.PhaseIModel(best["model_config"]["channel_counts"],best["model_config"]["use_experts"])
    model.load_state_dict(best["model_state"])
    monkeypatch.undo()
    metrics=evaluate(out,data,device="cpu",output_dir=tmp_path/"evaluation")
    assert metrics["test_events"]==4
    for name in ("test_predictions.npz","baseline_predictions.npz","metrics.json","report.md"):
        assert (tmp_path/"evaluation"/name).exists()
    arrays=np.load(tmp_path/"evaluation"/"baseline_predictions.npz")
    assert set(("particle","evt_id","energy_prior_mean","pid_mask_lookup")).issubset(arrays.files)


def test_requested_cuda_fails_fast_when_unavailable(tmp_path,monkeypatch):
    data=tmp_path/"data"; _tiny_dataset(data)
    monkeypatch.setattr(torch.cuda,"is_available",lambda:False)
    with pytest.raises(RuntimeError,match="CUDA device cuda:0 was requested"):
        train_module.train(data,tmp_path/"best.pt",epochs=1,device="cuda:0")
