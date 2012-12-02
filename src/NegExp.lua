local NegExp, parent = torch.class('nn.NegExp', 'nn.Module')

function NegExp:__init(inputSize)
	parent.__init(self)
end

function NegExp:updateOutput(input)
	self.output:resizeAs(input)
	self.output:copy(input * -1):exp()
	return self.output
end

function NegExp:updateGradInput(input, gradOutput)
	self.gradInput:resizeAs(input)
	self.gradInput:copy(input * -1):exp():mul(-1):cmul(gradOutput)
	return self.gradInput
end