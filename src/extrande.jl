struct Extrande <: AbstractAggregatorAlgorithm end

"""
Extrande sampling method for jumps with defined rate bounds.
"""

nullaffect!(integrator) = nothing
const NullAffectJump = ConstantRateJump((u,p,t) -> 0.0, nullaffect!)

mutable struct ExtrandeJumpAggregation{T,S,F1,F2,F3,F4,RNG} <: AbstractSSAJumpAggregator
  next_jump::Int
  prev_jump::Int
  next_jump_time::T
  end_time::T
  cur_rates::Vector{T}
  sum_rate::T
  ma_jumps::S
  rate_bnds::F3
  wds::F4
  rates::F1
  affects!::F2
  save_positions::Tuple{Bool,Bool}
  rng::RNG
end

function ExtrandeJumpAggregation(nj::Int, njt::T, et::T, crs::Vector{T}, sr::T, maj::S, rs::F1, affs!::F2, sps::Tuple{Bool,Bool}, rng::RNG; rate_bounds::F3, windows::F4, kwargs...) where {T,S,F1,F2,F3,F4,RNG}

  ExtrandeJumpAggregation{T,S,F1,F2,F3,F4,RNG}(nj, nj, njt, et, crs, sr, maj, rate_bounds, windows, rs, affs!, sps, rng)
end


############################# Required Functions ##############################
function aggregate(aggregator::Extrande, u, p, t, end_time, constant_jumps,
                   ma_jumps, save_positions, rng; bounded_va_jumps, kwargs...)

  display("hello")
  rates, affects! = get_jump_info_fwrappers(u, p, t, (constant_jumps..., bounded_va_jumps..., NullAffectJump))
  rbnds, wnds = get_va_jump_bound_info_fwrapper(u, p, t, bounded_va_jumps)


  build_jump_aggregation(ExtrandeJumpAggregation, u, p, t, end_time, ma_jumps,
                         rates, affects!, save_positions, rng; u=u, rate_bounds=rbnds, windows=wnds, kwargs...)
end

# set up a new simulation and calculate the first jump / jump time
function initialize!(p::ExtrandeJumpAggregation, integrator, u, params, t)
  p.end_time = integrator.sol.prob.tspan[2]
  generate_jumps!(p, integrator, u, params, t)
end

# execute one jump, changing the system state
@inline function execute_jumps!(p::ExtrandeJumpAggregation, integrator, u, params, t)
  # execute jump
  u = update_state!(p, integrator, u)
  nothing
end

# calculate the next jump / jump time
# tuple-based constant jumps
@fastmath function time_to_next_jump(p::ExtrandeJumpAggregation{T,S,F1,F2,F3,F4,RNG}, u, params, t) 
    where {T,S,F1 <: Tuple, F2 <: Tuple,F3,F4,RNG}
#@fastmath function time_to_next_jump(p::ExtrandeJumpAggregation, u, params, t) 
  prev_rate = zero(t)
  new_rate  = zero(t)
  cur_rates = p.cur_rates

  # mass action rates
  majumps   = p.ma_jumps
  idx       = get_num_majumps(majumps)
  @inbounds for i in 1:idx
    new_rate     = evalrxrate(u, i, majumps)
    cur_rates[i] = new_rate + prev_rate
    prev_rate    = cur_rates[i]
  end

  # constant & variable jump rates
  rates = p.rates
  if !isempty(rates)
    idx  += 1
    fill_cur_rates(u, params, t, cur_rates, idx, rates...)
    @inbounds for i in idx:length(cur_rates)
      cur_rates[i] = cur_rates[i] + prev_rate
      prev_rate    = cur_rates[i]
    end
  end


  # Calculate bounds
  bounds = p.rate_bnds
  Bs = zeros(T, length(bounds))
  if !isempty(bounds)
      fill_cur_rates(u,params,t,Bs,idx,bnds)
  end

  @fastmath B = p.bnd_rate(u, params, t, p.window)
  B, randexp(p.rng) / B
end


function generate_jumps!(p::ExtrandeJumpAggregation, integrator, u, params, t)
  p.sum_rate, ttnj = time_to_next_jump(p, u, params, t)

  U = rand(p.rng)
  @fastmath p.next_jump_time = t + ttnj
  if p.next_jump_time > t + p.window
    p.next_jump_time = t + p.window
    p.next_jump = length(p.cur_rates)
  else
    if p.cur_rates[end] ≥ U * p.sum_rate
      @inbounds p.next_jump = findfirst(x -> x ≥ U * p.sum_rate, p.cur_rates)
    else 
      p.next_jump = length(p.cur_rates)
    end
  end
  nothing
end

