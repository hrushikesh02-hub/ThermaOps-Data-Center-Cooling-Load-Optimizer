"""One entry point for the dashboard, data checks, training and evaluation."""
import argparse
import json
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='ThermaOps cooling operations')
    parser.add_argument('command',nargs='?',default='dashboard',
        choices=['dashboard','prepare','train','evaluate','research','test'])
    parser.add_argument('--port',type=int,default=8501,help='Local dashboard port')
    parser.add_argument('--state-dir',type=Path,help='Custom directory for operator history')
    parser.add_argument('--output-dir',type=Path,help='New model bundle directory for training')
    parser.add_argument('--tune',action='store_true',help='Tune candidate models with training-only cross-validation')
    args = parser.parse_args()
    if args.command=='dashboard':
        if not 1<=args.port<=65535: parser.error('--port must be between 1 and 65535')
        from api.app import create_app
        print(f'Operator dashboard: http://127.0.0.1:{args.port}',flush=True)
        create_app(args.state_dir).run(host='127.0.0.1',port=args.port,debug=False,threaded=True)
    elif args.command=='prepare':
        from ml.preprocessing.prepare import prepare_datasets
        print(json.dumps(prepare_datasets(),indent=2))
    elif args.command=='train':
        from ml.config import RESULTS_DIR
        from ml.models.train import train_bundle
        output = args.output_dir or RESULTS_DIR/'training'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        report = train_bundle(output,args.tune)
        print(json.dumps(report['selected'],indent=2))
        print(f'Model bundle saved to {output.resolve()}')
    elif args.command=='evaluate':
        from ml.evaluation.evaluate import evaluate_models
        print(json.dumps(evaluate_models(),indent=2))
    elif args.command=='research':
        from ml.evaluation.run_research import run_research
        run_research()
    elif args.command=='test':
        import unittest
        root = Path(__file__).resolve().parent
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(str(root/'tests'),top_level_dir=str(root)))
        raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__=='__main__':
    main()
