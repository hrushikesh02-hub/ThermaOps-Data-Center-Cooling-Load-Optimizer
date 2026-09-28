"""Versioned model/data lineage shared by inference, calibration and OOD fitting."""
from functools import lru_cache
import hashlib
import json
import pandas as pd
from ml.config import DATA_DIR, MODELS_DIR


@lru_cache(maxsize=1)
def load_model_profile():
    path=MODELS_DIR/'model_profile.json'
    if not path.exists():
        raise FileNotFoundError('No active model bundle. Run: python main.py train --output-dir ml/models/artifacts')
    profile=json.loads(path.read_text(encoding='utf-8'))
    if profile.get('version')!=1:
        raise ValueError('Unsupported model profile version')
    prefix=profile.get('training_validation_prefix_rows')
    if type(prefix) is not int or prefix<0:
        raise ValueError('Invalid model-profile training/validation boundary')
    for name in ['train','validation','test']:
        actual=hashlib.sha256((DATA_DIR/f'{name}.csv').read_bytes()).hexdigest()
        if actual!=profile['dataset_sha256'][name]:
            raise ValueError(f'Model-profile dataset mismatch: {name}. Retrain and validate before deployment.')
    selected=json.loads((MODELS_DIR/'selected_models.json').read_text(encoding='utf-8'))
    if selected!=profile['selected_models']:
        raise ValueError('Selected models do not match their training profile. Deploy a complete model bundle.')
    if set(profile['model_sha256'])!=set(selected.values()):
        raise ValueError('Model profile must identify all selected artifacts')
    for name,expected in profile['model_sha256'].items():
        path=(MODELS_DIR/name).resolve()
        if not path.is_relative_to(MODELS_DIR.resolve()):
            raise ValueError('Model path must stay within models directory')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError(f'Model artifact does not match its training profile: {name}')
    profile['revision']=hashlib.sha256(json.dumps(profile,sort_keys=True).encode()).hexdigest()[:16]
    return profile


@lru_cache(maxsize=1)
def model_frames():
    profile=load_model_profile()
    train=pd.read_csv(DATA_DIR/'train.csv')
    validation=pd.read_csv(DATA_DIR/'validation.csv')
    prefix=profile['training_validation_prefix_rows']
    if prefix>=len(validation):
        raise ValueError('A model profile must retain held-out validation observations')
    train=pd.concat([train,validation.iloc[:prefix]],ignore_index=True)
    validation=validation.iloc[prefix:].reset_index(drop=True)
    test=pd.read_csv(DATA_DIR/'test.csv')
    if not train.timestamp.max()<validation.timestamp.min() or not validation.timestamp.max()<test.timestamp.min():
        raise ValueError('Model profile requires disjoint chronological train, validation and test intervals')
    return {'train':train,'validation':validation,'test':test}


def profile_summary():
    profile=load_model_profile()
    frames=model_frames()
    return {'name':profile['name'],'revision':profile['revision'],
        'splits':{name:{'rows':len(frame),'from':frame.timestamp.min(),'to':frame.timestamp.max(),
            'seasons':sorted(frame.season.unique().tolist())} for name,frame in frames.items()},
        'test_used_for_training':False}
