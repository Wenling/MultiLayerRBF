--[[
Step1: We have a training set Y^1 ...Y^P whose component variables have zero mean (or have been centered).
Step2: compute the covariance matrix A = 1/P Sigma^P_{i=1} YiYi'
Step3: diagonalize the covariance matrix: A = Q'\Lambda Q
Step4: Construct the matrix Q^k whose rows are the eigenvectors of largest eigenvectors of A (a subset rows of Q).
PCA^k(Y) = Q^k Y
]]
whitening = {};

-- The function should transforms the 'N' original (normalized) features to the top 'k' principal components.
function whitening:whitenDatasets(train, test, k)
	-- generate a feature-matrix 'X' (of the trainingset)
	local X = torch.Tensor(train:features(), train:size()):fill(0)
	for i = 1, train:size() do
		X[{{}, i}] = train[i][1]
	end
	local u, s, v = torch.svd(X)
	local u_k = u:sub(1, train:features(), 1, k)
	local X_new = u_k:t() * X
--	u_k^T X * X^T u = \Lambda
--	print(X_new * X_new:t())
	
	local X2 = torch.Tensor(test:features(), test:size()):fill(0)
	for i = 1, test:size() do
		X2[{{}, i}] = test[i][1]
	end
	local X2_new = u_k:t() * X2

	--refill train and test with new features
	for i = 1, train:size() do
		train[i][1] = X_new[{{}, i}]
	end
	for i = 1, test:size() do
		test[i][1] = X2_new[{{}, i}]
	end
	--redefine the features() function
	function train:features() return k end
	function test:features() return k end
	--normalize train and test
	isolet:normalize(train,test)
	return train,test

end