using CUDACore, CUDA, Random, Printf, LinearAlgebra
include("gemmul8_port.jl")
const BB=7; const BV=2.0^BB
function sl(Ah,s); D=Vector{CuArray{Int8,2}}(undef,s); t=copy(Ah); for i in 1:s; t.*=BV; di=round.(t); D[i]=Int8.(di); t.-=di; end; D; end
function schemeI(dA,dB;s=8,T=10)
    σA=scale_exp(dA;dims=2); σB=scale_exp(dB;dims=1)
    DA=sl(dA.*(2.0.^(.-σA)),s); DB=sl(dB.*(2.0.^(.-σB)),s)
    M_,N_=size(dA,1),size(dB,2); S=CUDA.zeros(Float64,M_,N_); Cij=CUDA.zeros(Int32,M_,N_)
    for i in 1:s, j in 1:s; i+j<=T||continue; fill!(Cij,Int32(0)); gemmI8!('N','N',Int32(1),DA[i],DB[j],Int32(0),Cij); S.+=Float64.(Cij).*(2.0^(-BB*(i+j))); end
    (S.*(2.0.^σA)).*(2.0.^σB)
end
gt(f;nw=2,nr=6)=(for _ in 1:nw;f();CUDACore.synchronize();end;ts=Float64[];for _ in 1:nr;t0=time_ns();f();CUDACore.synchronize();push!(ts,(time_ns()-t0)/1e9);end;sort(ts)[cld(nr,2)])
function bench(M,N,K)
    rng=MersenneTwister(1); A=randn(rng,M,K); B=randn(rng,K,N); dA=CuArray(A); dB=CuArray(B)
    tf=gt(()->dA*dB); tI=gt(()->schemeI(dA,dB;s=8,T=10)); tII=gt(()->gemmul8_dgemm(A,B;num_moduli=14))
    @printf("  %d^3: FP64 %7.2f | SchemeI(43) %7.2f | GEMMul8-port(14) %7.2f ms  (%.2fx SchemeI, %.2fx FP64)\n", M, tf*1e3, tI*1e3, tII*1e3, tI/tII, tf/tII)
end
println("\n== GEMMul8 port benchmark ==")
for s in [1024,2048,4096]; bench(s,s,s); end
