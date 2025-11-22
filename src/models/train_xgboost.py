import argparse
'objective': 'reg:squarederror',
'max_depth': int(model_cfg.get('max_depth', 6)),
'eta': float(model_cfg.get('learning_rate', 0.05)),
'subsample': float(model_cfg.get('subsample', 0.8)),
'colsample_bytree': float(model_cfg.get('colsample_bytree', 0.8)),
'seed': int(model_cfg.get('random_state', seed))
}


evallist = [(dtrain, 'train'), (dtest, 'eval')]
num_round = int(model_cfg.get('n_estimators', 500))
bst = xgb.train(params, dtrain, num_round, evallist=evallist,
early_stopping_rounds=cfg['training'].get('early_stopping_rounds', 50),
verbose_eval=cfg['training'].get('verbose', False))


preds = bst.predict(dtest, ntree_limit=bst.best_ntree_limit if hasattr(bst, 'best_ntree_limit') else None)
rmse = sqrt(mean_squared_error(y_test, preds))
mae = mean_absolute_error(y_test, preds)
mapep = mape(y_test, preds)


results.append({'run': run, 'rmse': rmse, 'mae': mae, 'mape': mapep})


if rmse < best_rmse:
best_rmse = rmse
best_model = bst


# final train on full data
dtrain_full = xgb.DMatrix(X, label=y)
final_params = params
final_bst = xgb.train(final_params, dtrain_full, num_round)


os.makedirs(out_cfg.get('model_dir', 'models/'), exist_ok=True)
model_path = os.path.join(out_cfg.get('model_dir', 'models/'), f"xgboost_{datetime.now().strftime('%Y%m%d_%H%M%S')}.bst")
final_bst.save_model(model_path)


# write experiment logs
os.makedirs(os.path.dirname(out_cfg.get('logs_csv', 'logs/experiments.csv')), exist_ok=True)
logs_csv = out_cfg.get('logs_csv', 'logs/experiments.csv')
df_res = pd.DataFrame(results)
df_res['model'] = 'xgboost'
df_res.to_csv(logs_csv, index=False)


# metrics summary
os.makedirs(os.path.dirname(out_cfg.get('metrics_csv', 'results/metrics/baseline_vs_models.csv')), exist_ok=True)
metrics_csv = out_cfg.get('metrics_csv', 'results/metrics/baseline_vs_models.csv')
summary = pd.DataFrame([{'model': 'xgboost', 'rmse': float(np.mean([r['rmse'] for r in results])), 'mae': float(np.mean([r['mae'] for r in results])), 'mape': float(np.mean([r['mape'] for r in results]))}])
summary.to_csv(metrics_csv, index=False)


print(f"Saved final model to {model_path}")
print(f"Saved CV results to {logs_csv} and metrics summary to {metrics_csv}")




if __name__ == '__main__':
parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True)
args = parser.parse_args()
main(args.config)
