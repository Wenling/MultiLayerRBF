--[[

In general, all you need to do is to load this file with isolet.data
presented in you current directory as follows:
t7> dofille "isolet.lua"
then, you can split out shuffled and normalized training and testing data by
calling isolet:getDatasets(train_size,test_size), for example:
t7> train, test = isolet:getDatasets(3000,1000)

A dataset is an object which implements the operator dataset[index] and
implements the method dataset:size(). The size() methods returns the number of
examples and dataset[i] has to return the i-th example. An example has to be
an object which implements the operator example[field], where field often
takes the value 1 (for input features) or 2 (for corresponding labels), i.e
an example is a pair of input and output objects."

]]
dofile("whitening.lua")

-- the isolet dataset
isolet = {};
isolet_train = {};
isolet_test = {};

-- -- The dataset has 6238 training + 1559 testing samples
function isolet:size() return 7797 end
function isolet_train:size() return 6238 end
function isolet_test:size() return 1559 end

-- Each row (observaton) has 57 features
function isolet:features() return 617 end

-- We have 26 classes, where the digit i is class (i+1).
function isolet:classes() return 26 end

-- Read csv files from the isolet.data
function isolet:readFile()
	-- CSV reading using simple regular expression :)
	local train_file = '../assign3-dataset/isolet1+2+3+4.data'
	local tr_fp = assert(io.open (train_file))
	local train_csvtable = {}
	for line in tr_fp:lines() do
		local row = {}
		for value in line:gmatch("[^,]+") do
			-- note: doesn\'t work with strings that contain , values
			row[#row+1] = value
		end
		train_csvtable[#train_csvtable+1] = row
	end
	-- Generating random order
	local rorder = torch.randperm(isolet_train:size())
	-- iterate over rows
	for i = 1, isolet_train:size() do
		-- iterate over columns (1 .. num_features)
		local input = torch.Tensor(isolet:features())
		for j = 1, isolet:features() do
			-- set entry in feature matrix
			input[j] = train_csvtable[i][j]
		end
		-- get class label from last column (num_features+1)
		local output = train_csvtable[i][isolet:features()+1]
		output = output:sub(1, output:len() - 1)
		-- Shuffled dataset
		isolet_train[rorder[i]] = {input, tonumber(output)}
	end

	local test_file = '../assign3-dataset/isolet5.data'
	local te_fp = assert(io.open (test_file))
	local test_csvtable = {}
	for line in te_fp:lines() do
		local row = {}
		for value in line:gmatch("[^,]+") do
			-- note: doesn\'t work with strings that contain , values
			row[#row+1] = value
		end
		test_csvtable[#test_csvtable+1] = row
	end
	-- Generating random order
	rorder = torch.randperm(isolet_test:size())
	-- iterate over rows
	for i = 1, isolet_test:size() do
		-- iterate over columns (1 .. num_features)
		local input = torch.Tensor(isolet:features())
		for j = 1, isolet:features() do
			-- set entry in feature matrix
			input[j] = test_csvtable[i][j]
		end
		-- get class label from last column (num_features+1)
		local output = test_csvtable[i][isolet:features()+1]
		output = output:sub(1, output:len() - 1)
		-- Shuffled dataset
		isolet_test[rorder[i]] = {input, tonumber(output)}
	end
end

-- Split the dataset into two sets train and test
-- isolet:readFile() must have been executed
function isolet:split(train_size, test_size)
	local train = {}
	local test = {}
	function train:size() return train_size end
	function test:size() return test_size end
	function train:features() return isolet:features() end
	function test:features() return isolet:features() end
	-- iterate over rows
	for i = 1,train:size() do
		-- Cloning data instead of referencing, so that the datset can be split multiple times
		train[i] = {isolet_train[i][1]:clone(), isolet_train[i][2]}
	end
	-- iterate over rows
	for i = 1,test:size() do
		-- Cloning data instead of referencing
		test[i] = {isolet_test[i][1]:clone(), isolet_test[i][2]}
	end

	return train, test
end

-- Normalize the dataset using training set's mean and std
-- train and test must be returned from isolet:split
function isolet:normalize(train, test)
	-- Allocate mean and variance vectors
	local mean = torch.zeros(train:features())
	local var = torch.zeros(train:features())
	-- Iterative mean computation
	for i = 1,train:size() do
		mean = mean*(i-1)/i + train[i][1]/i
	end
	-- Iterative variance computation
	for i = 1,train:size() do
		var = var*(i-1)/i + torch.pow(train[i][1] - mean,2)/i
	end
	-- Get the standard deviation
	local std = torch.sqrt(var)
	-- If any std is 0, make it 1
	std:apply(function (x) if x == 0 then return 1 end end)
	-- Normalize the training dataset
	for i = 1,train:size() do
		train[i][1] = torch.cdiv(train[i][1]-mean, std)
	end
	
	-- Normalize the testing dataset
	local mean = torch.zeros(test:features())
	local var = torch.zeros(test:features())
	-- Iterative mean computation
	for i = 1,test:size() do
		mean = mean*(i-1)/i + test[i][1]/i
	end
	-- Iterative variance computation
	for i = 1,test:size() do
		var = var*(i-1)/i + torch.pow(test[i][1] - mean,2)/i
	end
	-- Get the standard deviation
	local std = torch.sqrt(var)
	-- If any std is 0, make it 1
	std:apply(function (x) if x == 0 then return 1 end end)
	-- Normalize the training dataset
	for i = 1,test:size() do
		test[i][1] = torch.cdiv(test[i][1]-mean, std)
	end

	return train, test
end

-- Get the train and test datasets
function isolet:getIsoletDatasets(train_size, test_size)
	-- If file not read, read the files
	if isolet[1] == nil then isolet:readFile() end
	-- Split the dataset
	local train, test = isolet:split(train_size, test_size)
	-- Normalize the dataset
	train, test = isolet:normalize(train, test)
	-- return train and test datasets
	return train, test
end

function testIsolet()
	-- test normalization: If the features in the two vectors are all in {-1, 0, 1} then the dataset is normalized correctly
	local tr, te = isolet:getIsoletDatasets(2,2)
	-- test whitening: u_k^T X * X^T u = diag matrix
	tr, te = whitening:whitenDatasets(tr, te, 1)
	print (te:features())
end
--testIsolet()