local RBFA, parent = torch.class('nn.RBFA', 'nn.Module')

function RBFA:__init(inputSize, outputSize)
   parent.__init(self)

   self.weight = torch.rand(outputSize*2, inputSize) *2-1
   self:normA()
   self.gradWeight = torch.Tensor(outputSize, inputSize)
   self.input_size=inputSize
   print(self.weight:size(1))
end


function RBFA:normA()
   for i=1,self.weight:size(1)/2 do
      self.weight[{{self.weight:size(1)/2+i},{}}]:mul(1/torch.norm(self.weight[{{self.weight:size(1)/2+i},{}}]))
   end
end

function RBFA:updateOutput(input)
   self:normA()
   if input:dim() == 1 then
      self.output:resize(self.weight:size(1)/2):zero()
      for i=1,self.weight:size(1)/2 do
         local tmp=input-self.weight:select(1,i)
         self.output[i]=tmp*torch.cmul(tmp,self.weight[{{self.weight:size(1)/2+i},{}}])
      end
   elseif input:dim() == 2 then
      local nframe = input:size(1)
      local nunit =self.weight:size(1)/2

      self.output:resize(nframe, nunit):zero()
      for k=1,nframe do
         for i=1,nunit do
            local tmp=input[k]-self.weight:select(1,i)
            self.output[k][i]=tmp*tmp
         end
      end
   else
      error('input must be vector or matrix')
   end

   return self.output
end

function RBFA:updateGradInput(input, gradOutput)
   if self.gradInput then
      local tmp=torch.Tensor(self.input_size, self.weight:size(1)/2)
      if input:dim() == 1 then
         self.gradInput:resizeAs(input)
         tmp:copy(self.weight[{{1,self.weight:size(1)/2},{}}]:t()*-2)
         for i=1,self.weight:size(1)/2 do
            tmp:select(2,i):add(2,input)
         end
         tmp:cmul(self.weight[{{self.weight:size(1)/2+1,self.weight:size(1)},{}}])
         self.gradInput:addmv(0, 1, tmp, gradOutput)
      elseif input:dim() == 2 then
         self.gradInput:resizeAs(input)
         local nframe = input:size(1)
         for k=1,nframe do
            tmp:copy(self.weight:t()*-2)
            for i=1,self.weight:size(1) do
              tmp:select(2,i):add(2,input[i])
            end
            self.gradInput[k]:addmv(0, 1,tmp,gradOutput[k])
         end
      end
      return self.gradInput
   end
end

function RBFA:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   if input:dim() == 1 then
      self.gradWeight[{{1,self.weight:size(1)/2},{}}]:addmm(scale,torch.diag(gradOutput),torch.cmul(self.weight[{{self.weight:size(1)/2+1,self.weight:size(1)},{}}],torch.addr(2,self.weight[{{1,self.weight:size(1)/2},{}}],-2,torch.ones(self.weight:size(1)/2),input)))
      self.gradWeight[{{self.weight:size(1)/2+1,self.weight:size(1)},{}}]:addmm(scale,torch.diag(gradOutput),torch.cmul(torch.addr(2,self.weight[{{1,self.weight:size(1)/2},{}}],-2,torch.ones(self.weight:size(1)/2),input),torch.addr(2,self.weight[{{1,self.weight:size(1)/2},{}}],-2,torch.ones(self.weight:size(1)/2),input)))
      --print(torch.norm(self.gradWeight))
   elseif input:dim() == 2 then
      local nframe = input:size(1)
      for k=1,nframe do
         self.gradWeight:addmm(scale,torch.diag(gradOutput[k]),torch.addr(2,self.weight,-2,torch.ones(self.weight:size(1)),input[k]))
      end
   end

end