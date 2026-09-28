"""Validate supplied raw/processed datasets and excluded model inputs."""
import json
from ml.config import DATA_DIR, RESULTS_DIR
from ml.preprocessing.validate import audit_data
from ml.utils import check_leakage
from ml.config import ALL_FEATURES, TARGETS


def prepare_datasets():
    paths = [DATA_DIR/f'{name}.csv' for name in ['train','validation','test']]
    missing = [path for path in paths if not path.exists()]
    if missing:
        names=', '.join(path.name for path in missing)
        raise FileNotFoundError(f'Missing required versioned processed dataset(s): {names}')
    report = audit_data()
    for target in TARGETS+['cooling_power_kw']:
        check_leakage(ALL_FEATURES,target)
    RESULTS_DIR.mkdir(parents=True,exist_ok=True)
    (RESULTS_DIR/'data_quality.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report
