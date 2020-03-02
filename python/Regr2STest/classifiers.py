from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble.forest import RandomForestClassifier


classifier_dict = {
    'Log. Regr.': LogisticRegression(penalty='none', solver='saga', max_iter=10000),
    'QDA': QuadraticDiscriminantAnalysis(),
    'NN': KNeighborsClassifier(),
    'XGBoost (d3, n1000)': XGBClassifier(n_estimators=1000),
    'XGBoost (d3, n100)': XGBClassifier(n_estimators=100),
    'XGBoost (d3, n500)': XGBClassifier(n_estimators=500),
    'XGBoost (d5, n1000)': XGBClassifier(max_depth=5, n_estimators=1000),
    'XGBoost (d5, n100)': XGBClassifier(max_depth=5, n_estimators=100),
    'XGBoost (d5, n500)': XGBClassifier(max_depth=5, n_estimators=500),
    'XGBoost (d10, n100)': XGBClassifier(max_depth=10),
    'XGBoost (d10, n500)': XGBClassifier(max_depth=10, n_estimators=500),
    'RF10': RandomForestClassifier(n_estimators=10),
    'RF100': RandomForestClassifier(n_estimators=100),
    'RF500': RandomForestClassifier(n_estimators=500),
    'RF1000': RandomForestClassifier(n_estimators=1000),
    'MLP': MLPClassifier(alpha=0, max_iter=1000),
    'Gauss_Proc1': GaussianProcessClassifier(RBF(1.0)),
    'Gauss_Proc2': GaussianProcessClassifier(RBF(.1)),
    'Gauss_Proc3': GaussianProcessClassifier(RBF(.5)),
    'Gauss_Proc4': GaussianProcessClassifier(0.5 * RBF(.1)),
    'MLP1t': MLPClassifier((32, 16), activation='tanh', alpha=0, max_iter=25000),
    'MLP1': MLPClassifier((32, 16), activation='relu', alpha=0, max_iter=25000),
    'MLP2t': MLPClassifier((64, 32, 32), activation='tanh', alpha=0, max_iter=25000),
    'MLP2': MLPClassifier((64, 32, 32), activation='relu', alpha=0, max_iter=25000),
    'MLP3t': MLPClassifier((128, 64, 32), activation='tanh', alpha=0, max_iter=25000),
    'MLP3': MLPClassifier((128, 64, 32), activation='relu', alpha=0, max_iter=25000),
    'MLP3t_a': MLPClassifier((128, 32, 32), activation='tanh', alpha=0, max_iter=25000),
    'MLP3_a': MLPClassifier((128, 32, 32), activation='relu', alpha=0, max_iter=25000),
    'MLP4t': MLPClassifier((128, 64, 32,  32), activation='tanh', alpha=0, max_iter=25000),
    'MLP4': MLPClassifier((128, 64, 32,  32), activation='relu', alpha=0, max_iter=25000),
    'MLP5t': MLPClassifier((128, 64, 64, 32), activation='tanh', alpha=0, max_iter=25000),
    'MLP5': MLPClassifier((128, 64, 64, 32), activation='relu', alpha=0, max_iter=25000),
    'MLP6t': MLPClassifier((256, 128, 64, 32), activation='tanh', alpha=0, max_iter=25000),
    'MLP6': MLPClassifier((256, 128, 64, 32), activation='relu', alpha=0, max_iter=25000),
    'MLP7': MLPClassifier((512, 256, 64, 32, 32), activation='relu', alpha=0, max_iter=25000),
    'MLP7t': MLPClassifier((512, 256, 64, 32, 32), activation='tanh', alpha=0, max_iter=25000),
    'MLP8': MLPClassifier((1024, 512, 256, 64, 32, 32), activation='relu', alpha=0, max_iter=25000),
    'MLP8t': MLPClassifier((1024, 512, 256, 64, 32, 32), activation='tanh', alpha=0, max_iter=25000)
}
