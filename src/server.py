import flwr as fl
import xgboost as xgb
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

# Global Test Set load karna
test_data = np.load("data/processed/global_test.npz")
X_test, y_test = test_data["X"], test_data["y"]
dtest = xgb.DMatrix(X_test, label=y_test)

class FedXGBStrategy(fl.server.strategy.FedAvg):
    """Custom strategy jo XGBoost JSON models ko safely handle aur evaluate karti hai."""
    
    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        
        # Har client se aaye hue XGBoost JSON bytes ko save karna
        for idx, (_, fit_res) in enumerate(results):
            model_bytes = fit_res.parameters.tensors[0]
            with open("models/global_xgb_model.json", "wb") as f:
                f.write(model_bytes)
        
        print(f"[SERVER] Round {server_round} aggregation complete. Global model updated!")
        return results[0][1].parameters, {}

    def evaluate(self, server_round, parameters):
        """Override default FedAvg evaluate taaki JSON bytes ko NumPy decode na kare."""
        if server_round == 0:
            return 0.0, {}
        
        try:
            bst = xgb.Booster()
            bst.load_model("models/global_xgb_model.json")
            preds = bst.predict(dtest)
            pred_labels = (preds > 0.5).astype(int)
            
            f1 = float(f1_score(y_test, pred_labels))
            auc = float(roc_auc_score(y_test, preds))
            print(f"[SERVER ROUND {server_round}] Global Test F1-Score: {f1:.4f} | ROC-AUC: {auc:.4f}")
            return 1.0 - f1, {"f1_score": f1, "roc_auc": auc}
        except Exception as e:
            print(f"[SERVER EVAL ERROR] {e}")
            return 0.0, {}

def main():
    print("==========================================")
    print("  FEDERATED XGBOOST CENTRAL SERVER START  ")
    print("==========================================")
    
    strategy = FedXGBStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
    )

    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )
    print("\n[SUCCESS] Federated Training Finished! Final model saved at 'models/global_xgb_model.json'")

if __name__ == "__main__":
    main()
