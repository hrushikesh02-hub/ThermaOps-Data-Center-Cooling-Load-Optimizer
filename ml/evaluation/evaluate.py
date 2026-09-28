"""Evaluate the active model bundle without modifying model selection."""
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ml.config import ALL_FEATURES, RESULTS_DIR
from ml.models.profile import model_frames, profile_summary
from ml.models.train import TARGET_KEYS, metrics
from ml.optimization.fitness import get_surrogate_models
from ml.preprocessing.validate import audit_data


def evaluate_models():
    audit_data()
    frames = model_frames()
    output = RESULTS_DIR/'evaluation'
    output.mkdir(parents=True,exist_ok=True)
    report = {'profile':profile_summary(),'models':{}}
    rows, importances = [], []
    for (key,target), model in zip(TARGET_KEYS.items(),get_surrogate_models()):
        result = {'target':target,'estimator':type(model.named_steps['model']).__name__}
        for split in ['validation','test']:
            frame = frames[split]
            pred = model.predict(frame[ALL_FEATURES])
            result[split] = metrics(frame[target],pred)
            rows.append({'model':key,'target':target,'split':split,**result[split]})
            if split=='validation':
                result['validation_upper_residual_95'] = float(max(0,np.quantile(frame[target].to_numpy()-pred,.95)))
            else:
                fig,axes = plt.subplots(1,2,figsize=(11,4))
                axes[0].scatter(frame[target],pred,alpha=.3,s=8)
                limits = [min(frame[target].min(),pred.min()),max(frame[target].max(),pred.max())]
                axes[0].plot(limits,limits,'r--')
                axes[0].set(xlabel='Recorded value',ylabel='Predicted value',title=target)
                axes[1].hist(frame[target].to_numpy()-pred,bins=40)
                axes[1].set(xlabel='Recorded minus predicted',ylabel='Records',title='Prediction errors')
                fig.tight_layout();fig.savefig(output/f'{key}_accuracy.png',dpi=150);plt.close(fig)
        estimator = model.named_steps['model']
        if hasattr(estimator,'feature_importances_'):
            names = model.named_steps['preprocessor'].get_feature_names_out()
            imp = pd.DataFrame({'feature':names,'importance':estimator.feature_importances_}).sort_values('importance',ascending=False)
            importances.extend({'model':key,**row} for row in imp.to_dict('records'))
            fig,axis = plt.subplots(figsize=(9,6))
            top = imp.head(15).iloc[::-1]
            axis.barh(top.feature,top.importance);axis.set_title(f'{target}: feature importance')
            fig.tight_layout();fig.savefig(output/f'{key}_feature_importance.png',dpi=150);plt.close(fig)
        report['models'][key] = result
    pd.DataFrame(rows).to_csv(output/'model_metrics.csv',index=False)
    pd.DataFrame(importances).to_csv(output/'feature_importance.csv',index=False)
    (output/'model_metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return report
