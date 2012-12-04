require("nn")
dofile("isolet.lua")
dofile("whitening.lua")
dofile("train.lua")
dofile("test.lua")
dofile("RBF.lua")
dofile("RBFA.lua")
dofile("MulPos.lua")
dofile("NegExp.lua")
dofile("LinearReg.lua")
dofile("models.lua")

local function Q4()  
	local train_size = tonumber(arg[1])
	local test_size = 1500
	local testFunc = tonumber(arg[2])
	local k = tonumber(arg[3])
	local HU = tonumber(arg[4])
	local learningRate = tonumber(arg[5])
	local initW = tonumber(arg[6])
	local epoch = tonumber(arg[7])
	local lambda = tonumber(arg[8])
	local outputs = 26
	
	local trainset, testset = isolet:getIsoletDatasets(train_size,test_size)
	whitening:whitenDatasets(trainset, testset, k)
	
	local err = 1
	if (testFunc == 1) then
		err = testMod(trainLogReg(trainset, trainset:features(), outputs), testset)
	end
	if (testFunc == 2) then
		err = testMod(trainTwoLinReg(trainset, trainset:features(), HU, 26, learningRate, epoch), testset)
	end
	if (testFunc == 3) then
		err = testMod(trainRBF(trainset, trainset:features(), HU, 26, initW, learningRate, epoch), testset)
	end
	if (testFunc == 4) then
		err = testMod(trainAntiRBF(trainset, trainset:features(), HU, 26, initW, learningRate, epoch), testset)
	end
	if (testFunc == 5) then
		local lambda = 0.5
		local interOutputs = 13
		local interHU = 80
		local W2 = -2
		err = testMod(trainHRBF(trainset, trainset:features(), lambda, HU, interOutputs, interHU, outputs, initW, W2, learningRate, epoch), testset)
	end
	if (testFunc == 6) then
		err = testMod(trainMulLinReg(trainset, trainset:features(), HU, 26, learningRate, epoch), testset)
	end
	if (testFunc == 7) then
		err = testMod(trainTwoLinReg2(trainset, trainset:features(), HU, 26, learningRate, epoch, lambda), testset)
	end
	if (testFunc == 8) then
		err = testMod(trainMulLinReg2(trainset, trainset:features(), HU, 26, learningRate, epoch, lambda), testset)
	end
	print(k .. "," .. HU .. "," .. learningRate .. "," .. initW .. "," .. epoch .. "," ..lambda .. "," .. err)
end

Q4()
--Q36()

-- mlp=nn.Parallel(2,1);     -- iterate over dimension 2 of input
-- mlp:add(nn.Linear(10,3)); -- apply to first slice
-- mlp:add(nn.Linear(10,2))  -- apply to first second slice
-- print(mlp:forward(torch.randn(10,2)))