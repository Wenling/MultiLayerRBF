
function trainLogReg(dataset, inputs, outputs)
	local mlp = modLogReg(inputs, outputs)
	local criterion = nn.ClassNLLCriterion()
	local trainer = nn.StochasticGradient(mlp, criterion)
	trainer:train(dataset)
	return mlp
end

function trainTwoLinReg(dataset, inputs, HU, outputs, learningRate, epoch)
	local mlp = modTwoLinReg(inputs, HU, outputs)
	local criterion = nn.ClassNLLCriterion()
	local trainer = nn.StochasticGradient(mlp, criterion)
	trainer.learningRate = learningRate
	trainer.maxIteration = epoch
	trainer:train(dataset)
	return mlp
end

function trainTwoLinReg2(dataset, inputs, HU, outputs, learningRate, epoch, lambda)
	local mlp = modTwoLinReg2(inputs, HU, outputs, lambda)
	local criterion = nn.ClassNLLCriterion()
	local trainer = nn.StochasticGradient(mlp, criterion)
	trainer.learningRate = learningRate
	trainer.maxIteration = epoch
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

	for j = 1, epoch do
	local shuffledIndices = torch.randperm(dataset:size(), 'torch.LongTensor')
		for i = 1,dataset:size() do
	
			-- feed it to the neural network and the criterion
			criterion:forward(mlp:forward(dataset[shuffledIndices[i]][1]), dataset[shuffledIndices[i]][2])
	
			-- train over this example in 3 steps
			-- (1) zero the accumulation of the gradients
			mlp:zeroGradParameters()
			-- (2) accumulate gradients
			mlp:backward(dataset[shuffledIndices[i]][1], criterion:backward(mlp.output, dataset[shuffledIndices[i]][2]))
			-- (3) update parameters with a 0.01 learning rate
			mlp:updateParameters(learningRate)
		end	
	end
	--	local trainer = nn.StochasticGradient(mlp, criterion)
	--	trainer.learningRate = learningRate
	--	trainer.maxIteration = epoch
	--	trainer:train(dataset)
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
	for j = 1, epoch do
	local shuffledIndices = torch.randperm(dataset:size(), 'torch.LongTensor')
		for i = 1,dataset:size() do
	
			-- feed it to the neural network and the criterion
			criterion:forward(mlp:forward(dataset[shuffledIndices[i]][1]), dataset[shuffledIndices[i]][2])
	
			-- train over this example in 3 steps
			-- (1) zero the accumulation of the gradients
			mlp:zeroGradParameters()
			-- (2) accumulate gradients
			mlp:backward(dataset[shuffledIndices[i]][1], criterion:backward(mlp.output, dataset[shuffledIndices[i]][2]))
			-- (3) update parameters with a 0.01 learning rate
			mlp:updateParameters(learningRate)
		end	
	end
	return mlp
end

function trainMulLinReg(dataset, inputs, HU, outputs, learningRate, epoch)
	local mlp = modMulLinReg(inputs, HU, outputs)
	local criterion = nn.ClassNLLCriterion()
	local trainer = nn.StochasticGradient(mlp, criterion)
	trainer.learningRate = learningRate
	trainer.maxIteration = epoch
	trainer:train(dataset)
	return mlp
end

function trainMulLinReg2(dataset, inputs, HU, outputs, learningRate, epoch, lambda)
	local mlp = modMulLinReg(inputs, HU, outputs, lambda)
	local criterion = nn.ClassNLLCriterion()
	local trainer = nn.StochasticGradient(mlp, criterion)
	trainer.learningRate = learningRate
	trainer.maxIteration = epoch
	trainer:train(dataset)
	return mlp
end