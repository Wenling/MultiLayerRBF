
-- test error rate of model
function testMod(model, dataset)
	local error = 0
	for i = 1,dataset:size() do
		local y = model:forward(dataset[i][1])
		local tmp, index = torch.max(y,1)
		if index[1] == dataset[i][2] then
			error = error / i * (i - 1)
		else
			error = error / i * (i - 1) + 1 / i
		end
	end

	return error
end

function Q36()
	local mytester = torch.Tester()
	local jac
		local ini = math.random(50,70)
		local inj = math.random(50,70)
		local moduleName = nn.RBFA(ini, inj)
		local input = torch.Tensor(ini):zero()
--	local ini = math.random(10,20)
--	local inj = math.random(10,20)
--	local ink = math.random(10,20)
--	local input = torch.Tensor(ini,inj,ink):zero()
--	local moduleName = nn.MulPos(ini*inj*ink, 1)
--	local moduleName = nn.NegExp()

	local precision = 1e-5
	local expprecision = 1e-4

	local nntest = {}
	local nntestx = {}

	-- Module differentiation testing
	function nntest.modDiffTest()
		local module = moduleName

		-- 1D
		local err = jac.testJacobian(module,input)
		mytester:assertlt(err,precision, 'error on state ')

		local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight)
		mytester:assertlt(err,precision, 'error on weight ')

		local err = jac.testJacobianUpdateParameters(module, input, module.weight)
		mytester:assertlt(err,precision, 'error on weight [direct update] ')

		for t,err in pairs(jac.testAllUpdate(module, input, 'weight', 'gradWeight')) do
			--			print(t)
			mytester:assertlt(err, precision, string.format(
			'error on weight [%s]', t))
		end


		-- 2D
		--		local nframe = math.random(50,70)
		--		local input = torch.Tensor(nframe, ini):zero()
		--
		--		local err = jac.testJacobian(module,input)
		--		mytester:assertlt(err,precision, 'error on state ')
		--
		--		local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight)
		--		mytester:assertlt(err,precision, 'error on weight ')
		--
		--		local err = jac.testJacobianUpdateParameters(module, input, module.weight)
		--		mytester:assertlt(err,precision, 'error on weight [direct update] ')
		--
		--		for t,err in pairs(jac.testAllUpdate(module, input, 'weight', 'gradWeight')) do
		--			mytester:assertlt(err, precision, string.format(
		--			'error on weight [%s]', t))
		--		end


		-- IO
		local ferr,berr = jac.testIO(module,input)
		mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
		mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')

	end

	mytester:add(nntest)

	if not nn then
		require 'nn'
		jac = nn.Jacobian
		mytester:run()
	else
		jac = nn.Jacobian
		mytester:run()
		--		function nn.test(tests)
		--			-- randomize stuff
		--			math.randomseed(os.time())
		--			mytester:run(tests)
		--		end
	end
end
--Q36()