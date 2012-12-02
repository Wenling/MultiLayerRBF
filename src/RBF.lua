-- The function is: z_i = sum_j( (x_j - W_ij)^2)

local RBF, parent = torch.class('nn.RBF', 'nn.Module')

function RBF:__init(inputSize, outputSize)
	parent.__init(self)

	self.weight = torch.Tensor(outputSize, inputSize)
--	self.weight = torch.rand(outputSize, inputSize) * 2 - 1;
	self.gradWeight = torch.Tensor(outputSize, inputSize)
	self.inputSize = inputSize
	self.outputSize = outputSize
	self:reset()
end

function RBF:reset(stdv)
	if stdv then
		stdv = stdv * math.sqrt(3)
	else
		stdv = 1./math.sqrt(self.weight:size(2))
	end

	-- we do this so the initialization is exactly
	-- the same than in previous torch versions
	for i=1,self.weight:size(1) do
		self.weight:select(1, i):apply(function()
			return torch.uniform(-stdv, stdv)
		end)
	end
end

function RBF:updateOutput(input)
	if input:dim() == 1 then
		self.output:resize(self.outputSize)
		local X = torch.Tensor(self.outputSize, input:size(1)):zero()
		X:addr(torch.ones(self.outputSize), input)
		for i = 1, self.outputSize do
			self.output[i] = torch.norm(X[i] - self.weight[i])^2
		end

	elseif input:dim() == 2 then
		local nframe = input:size(1)
		local nunit = self.outputSize

		self.output:resize(nframe, nunit)
		for i = 1, nframe do
			for j = 1, nunit do
				self.output[i][j] = torch.norm(input[i] - self.weight[j])^2
			end
		end
	else
		error('input must be vector or matrix')
	end

	return self.output
end

function RBF:updateGradInput(input, gradOutput)
	if self.gradInput then

		if input:dim() == 1 then
			self.gradInput:resizeAs(input)
			self.gradInput:addmv(0, 1, torch.addr(-2, self.weight, 2, torch.ones(self.outputSize), input):t(), gradOutput)

		elseif input:dim() == 2 then
			self.gradInput:resizeAs(input)
			local nframe = input:size(1)
			for i = 1, nframe do
				self.gradInput[i]:addmv(0, 1, torch.addr(-2, self.weight, 2, torch.ones(self.outputSize), input[i]):t(), gradOutput[i])
			end
		end

		return self.gradInput
	end
end

function RBF:accGradParameters(input, gradOutput, scale)
	scale = scale or 1

	if input:dim() == 1 then
		self.gradWeight:addmm(scale, torch.diag(gradOutput), torch.addr(2, self.weight, -2, torch.ones(self.outputSize), input))
--		print(torch.norm(self.gradWeight))
	elseif input:dim() == 2 then
		local nframe = input:size(1)
		for i = 1, nframe do
			self.gradWeight:addmm(scale, torch.diag(gradOutput[i]), torch.addr(2, self.weight, -2, torch.ones(self.outputSize), input[i]))
		end
	end

end