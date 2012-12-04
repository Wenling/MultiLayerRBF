local LinearReg, parent = torch.class('nn.LinearReg', 'nn.Module')

function LinearReg:__init(inputSize, outputSize, lambda)
   parent.__init(self)

   self.weight = torch.Tensor(outputSize, inputSize)
   self.bias = torch.Tensor(outputSize)
   self.gradWeight = torch.Tensor(outputSize, inputSize)
   self.gradBias = torch.Tensor(outputSize)
   self.lambda = torch.Tensor({lambda})
   
   self:reset()
end

function LinearReg:reset(stdv)
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
      self.bias[i] = torch.uniform(-stdv, stdv)
   end
end

function LinearReg:updateOutput(input)
   if input:dim() == 1 then
      self.output:resize(self.bias:size(1))
      self.output:copy(self.bias)
      self.output:addmv(1, self.weight, input)
   elseif input:dim() == 2 then
      local nframe = input:size(1)
      local nunit = self.bias:size(1)

      self.output:resize(nframe, nunit)
      self.output:zero():addr(1, input.new(nframe):fill(1), self.bias)
      self.output:addmm(1, input, self.weight:t())
   else
      error('input must be vector or matrix')
   end

   return self.output
end

function LinearReg:updateGradInput(input, gradOutput)
   if self.gradInput then

      if input:dim() == 1 then
         self.gradInput:resizeAs(input)
         self.gradInput:addmv(0, 1, self.weight:t(), gradOutput)
      elseif input:dim() == 2 then
         self.gradInput:resizeAs(input)
         self.gradInput:addmm(0, 1, gradOutput, self.weight)
      end

      return self.gradInput
   end
end

function LinearReg:accGradParameters(input, gradOutput, scale)
   scale = scale or 1

   if input:dim() == 1 then
      self.gradWeight:addr(scale, gradOutput, input)
      self.gradWeight:add(scale * self.lambda[1], self.weight)
      self.gradBias:add(scale, gradOutput + self.gradBias * 2 * self.lambda[1])      
   elseif input:dim() == 2 then
      local nframe = input:size(1)
      local nunit = self.bias:size(1)

      self.gradWeight:addmm(scale, gradOutput:t(), input)
      self.gradBias:addmv(scale, gradOutput:t(), input.new(nframe):fill(1))
   end

end

-- we do not need to accumulate parameters when sharing
LinearReg.sharedAccUpdateGradParameters = LinearReg.accUpdateGradParameters