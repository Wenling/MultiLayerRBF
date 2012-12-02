local MulPos, parent = torch.class('nn.MulPos', 'nn.Module')

function MulPos:__init(inputSize, W)
	parent.__init(self)

	self.weight = torch.Tensor({W})
	self.gradWeight = torch.Tensor(1)

	-- state

--	self:reset()
end

function MulPos:updateOutput(input)
	self.output:resizeAs(input)
	self.output:copy(input);
	self.output:mul(torch.exp(self.weight[1]));
	return self.output
end

function MulPos:updateGradInput(input, gradOutput)
	self.gradInput:resizeAs(input)
	self.gradInput:ones(input:size())
	self.gradInput:mul(torch.exp(self.weight[1])):cmul(gradOutput)
	return self.gradInput
end

function MulPos:accGradParameters(input, gradOutput, scale)
	scale = scale or 1
	self.gradWeight[1] = self.gradWeight[1] + scale * torch.dot(gradOutput, input * torch.exp(self.weight[1]));
end