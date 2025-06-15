import shap
import matplotlib.pyplot as plt

def explain_model(model, X_train, feature_names):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    # Summary plot
    shap.summary_plot(shap_values, X_train, feature_names=feature_names, show=False)
    plt.savefig("shap_summary.png")
    print("SHAP summary plot saved as 'shap_summary.png'")

    # Bar plot
    shap.summary_plot(shap_values, X_train, feature_names=feature_names, plot_type="bar", show=False)
    plt.savefig("shap_bar.png")
    print("SHAP bar plot saved as 'shap_bar.png'")
