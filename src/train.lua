
function trainLogReg(dataset, inputs, outputs)
	local mlp = modLogReg(inputs, outputs)
	local criterion = nn.ClassNLLCriterion()
	local trainer = nn.StochasticGradient(mlp, criterion)
	trainer:train(dataset)
	return mlp
end

function trainTwoLinReg(dataset, inputs, HU, outputs, learningRate)
	local mlp = modTwoLinReg(inputs, HU, outputs)
	local criterion = nn.ClassNLLCriterion()
	local trainer = nn.StochasticGradient(mlp, criterion)
	trainer.learningRate = learningRate
	trainer:train(dataset)
	return mlp
end

function trainRBF(dataset, inputs, HU, outputs, W, learningRate, epoch)
	local mlp = modRBF(inputs, HU, outputs, W)
	local criterion = nn.ClassNLLCriterion()
	local trainer = nn.StochasticGradient(mlp, criterion)
	trainer.learningRate = learningRate
	trainer.maxIteration = epoch
	trainer:train(dataset)
	return mlp
end

function trainAntiRBF(dataset, inputs, HU, outputs, W, learningRate, epoch)
	local mlp = modAntiRBF(inputs, HU, outputs, W)
	local criterion = nn.ClassNLLCriterion()
	local trainer = nn.StochasticGradient(mlp, criterion)
	trainer.learningRate = learningRate
	trainer.maxIteration = epoch
	trainer:train(dataset)
	return mlp
end

function trainRBFA(dataset, inputs, HU, outputs, W, learningRate, epoch)
	local mlp = modRBFA(inputs, HU, outputs, W)
	local criterion = nn.ClassNLLCriterion()
	local trainer = nn.StochasticGradient(mlp, criterion)
	trainer.learningRate = learningRate
	trainer.maxIteration = epoch
	trainer:train(dataset)
	return mlp
end

function trainHRBF(dataset, inputs, lambda, HU, interOutputs, interHU, outputs, W1, W2, learningRate, epoch)
	local mlp = modHRBF(inputs, lambda, HU, interOutputs, interHU, outputs, W1, W2)
	local criterion = nn.ClassNLLCriterion()
	local trainer = nn.StochasticGradient(mlp, criterion)
	trainer.learningRate = learningRate
	trainer.maxIteration = epoch
	trainer:train(dataset)
	return mlp
end