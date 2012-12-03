
-- Logistic regression module: f(x) = (exp(w^T x) - 1)/(exp(w^T x) + 1)
-- inputs: dimension of inputs, outputs: dimention of outputs of the first layer
function modLogReg(inputs, outputs)
	local logReg = nn.Sequential()

	logReg:add(nn.Linear(inputs, outputs))
	logReg:add(nn.Tanh())
	logReg:add(nn.LogSoftMax())
	return logReg
end

-- A two-layer neural network
-- inputs: dimension of inputs, HU: hidden unit, outputs: dimention of outputs
function modTwoLinReg(inputs, HU, outputs)
	local twoLinReg = nn.Sequential()

	twoLinReg:add(nn.Linear(inputs, HU))
	twoLinReg:add(nn.Tanh())
	twoLinReg:add(nn.Linear(HU, outputs))
	twoLinReg:add(nn.LogSoftMax())
	return twoLinReg
end

-- RBF model
-- inputs: dimension of inputs, HU: hidden unit, outputs: dimention of outputs, W: scalar
-- f_i(x) = sum(a_i exp(-b (x - c_i)^2))
function modRBF(inputs, HU, outputs, W)
	local rbf = nn.Sequential()

	rbf:add(nn.RBF(inputs, HU))
	rbf:add(nn.MulPos(HU, W))
	rbf:add(nn.NegExp())
	rbf:add(nn.Linear(HU, outputs))
	rbf:add(nn.LogSoftMax())
	return rbf
end

-- inputs: dimension of inputs, HU: hidden unit, outputs: dimention of outputs, W: scalar
-- f_i(x) = (x - c_i)^2
function modAntiRBF(inputs, HU, outputs, W)
	local rbf = nn.Sequential()

	rbf:add(nn.RBF(inputs, outputs))
	rbf:add(nn.LogSoftMax())
	return rbf
end

-- RBFA model
-- inputs: dimension of inputs, HU: hidden unit, outputs: dimention of outputs, W: scalar
-- f_i(x) = sum(a_i exp(-b Ai(x - c_i)^2))
function modRBFA(inputs, HU, outputs, W)
	local rbf = nn.Sequential()

	rbf:add(nn.RBFA(inputs, HU))
	rbf:add(nn.MulPos(HU, W))
	rbf:add(nn.NegExp())
	rbf:add(nn.Linear(HU, outputs))
	rbf:add(nn.LogSoftMax())
	return rbf
end

-- hierachical RBF model
-- inputs: dimension of inputs, inputs2: dimension of inputs2, HU: hidden unit, outputs: dimention of outputs, W: scalar
-- f_i(x) = sum(a_i exp(-b (x - c_i)^2))
function modHRBF(inputs, lambda,  HU, interOutputs, interHU, outputs, W1, W2)
	local rbf = nn.Sequential()
	local c = nn.Parallel(1, 2)
	local t = nn.Sequential()
	t:add(nn.Linear(inputs, outputs))
	c:add(t)
	rbf:add(c)

--	local rbfL1 = nn.Sequential()
--	rbfL1:add(nn.RBF(inputs * lambda, HU))
--	rbfL1:add(nn.MulPos(HU, W1))
--	rbfL1:add(nn.NegExp())
--	rbfL1:add(nn.Linear(HU, interOutputs))
--	c:add(rbfL1)
--	
--	local L1 = nn.Sequential()
--	L1:add(nn.Add(inputs - inputs * lambda, 0))
--	c:add(L1)
--	
--	rbf:add(c)
--	rbf:add(nn.RBF(interOutputs + inputs - inputs * lambda, interHU))
--	rbf:add(nn.MulPos(interHU, W2))
--	rbf:add(nn.NegExp())
--	rbf:add(nn.Linear(interHU, outputs))
--	rbf:add(nn.LogSoftMax())
	return rbf
end