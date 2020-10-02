import math
from sgd_D import SGDClassifier

def read_dataset(index):
	with open('problem_D_tests/' + index + '.txt', 'r') as inf:
		m = int(inf.readline())
		n = int(inf.readline())
		X_train, y_train = [], []
		for i in range(n):
			obj = list(map(int, inf.readline().split()))
			X_train.append(obj[:-1])
			y_train.append(obj[-1])
		n = int(inf.readline())
		X_test, y_test = [], []
		for i in range(n):
			obj = list(map(int, inf.readline().split()))
			X_test.append(obj[:-1])
			y_test.append(obj[-1])

	return X_train, y_train, X_test, y_test


def count_nrmse(y_predicted, y_test):
	try:
		return math.sqrt(sum((y_pred - y_t) ** 2 for y_pred, y_t in zip(y_predicted, y_test))) / len(y_test) / (
				max(y_test) - min(y_test))
	except:
		return 1

def count_smape(y_predicted, y_test):
	try:
		return sum(2 * abs(y_pred-y_t) / (abs(y_pred) + abs(y_t)) for y_pred, y_t in zip(y_predicted, y_test)) / len(y_test)
	except:
		return 2


def hyperparameter_optimization(X_train, y_train, X_test, y_test, Model):
	best_nrmse = 2
	for lr in [1e-20, 1e-18, 1e-15, 1e-12]:
		for reg_coef in [0.7, 0.9, 1]:
			params = [0, lr, reg_coef]
			model = Model(0, lr, reg_coef)
			model.fit(X_train, y_train, process_steps = False)
			for i in range(100, 100000, 100):
				for j in range(i - params[0]):
					model.process_step()
				y_predicted = model.predict(X_test)
				params[0] = i
				nrmse = count_smape(y_predicted, y_test)
				if nrmse < best_nrmse:
					best_nrmse = nrmse
					best_params = params
					print("Finished with best params", best_params, nrmse)
	print('Overall best params:', best_params)
	print('Overall best nrmse:', best_nrmse)
	return best_params, best_nrmse


if __name__ == "__main__":
	X_train, y_train, X_test, y_test = read_dataset('0.62_0.80')
	hyperparameter_optimization(X_train, y_train, X_test, y_test, SGDClassifier)
