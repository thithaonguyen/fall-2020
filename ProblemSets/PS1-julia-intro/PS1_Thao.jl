using Pkg;
Pkg.add.(["JLD2", "LinearAlgebra", "Random", "Statistics","CSV", "DataFrames", "FreqTables","Distributions"]);
using JLD2, LinearAlgebra, Random, Statistics, CSV, DataFrames, FreqTables, Distributions;
Random.seed!(1234)

using JLD
function q1()
    A = rand(Uniform(-5,10),10,7)
    B = rand(Normal(-2,15), 10, 7)
    C = zeros(5,7)
    for i in 1:5, j in 1:5
        C[i,j]= A[i,j]
        print(C[i,j])
    end
    C[:,6]=B[1:5,6]
    C[:,7]=B[1:5,7]
    D = zeros(10,7)
    for i in 1:10, j in 1:7
        if A[i,j] <=0
            D[i,j]= A[i,j]
            else D[i,j]= 0
        end
    E = reshape(B,(length(B),1))
    # Creating the F matrix
    F = zeros(10,7,2)
    F[:,:,1]=A
    F[:,:,2]=B
    F=permutedims(F,[3,1,2])  #Another way to create F matrix: F = cat(A,B, dims=3)
    G = kron(B,C)
    save("./matrixpractice.jld", "A", A, "B", B, "C", C, "D", D, "E", E, "F", F, "G", G) 
    save("./firstmatrix.jld", "A", A, "B", B, "C", C, "D", D)   
        
    df = DataFrame(C)
    CSV.write("./Cmatrix.csv",df)
            # CSV.write("/Users/thithaonguyen/Dropbox/Cmatrix.csv",df) ### write to another location  
    
    end
    # Another way to create D: D = A, D[D.>0].=0
    return A, B, C, D
end

A,B,C,D = q1()


function q2(A,B,C)
    AB=A.*B
    Cprime = reshape(C, (length(C),1))
    Cprime = Vector{Float64}()
        for i in 1:35
        if -5 ≤ C[i] ≤ 5
        append!(Cprime, C[i])
        end
        end
    print(Cprime)
    N = 15169;
    K = 6;
    T = 5;
    X = Array{Float64}(undef, N, K, T);
        for t in 1:T
            X[:,1,t]=ones(N)
            X[:,2,t]= rand(Bernoulli(0.75*(6-t)/5),N)
            X[:,3,t] = rand(Normal(15+t-1, 5*(t-1)),N)
            X[:,4,t] = rand(Normal(π*(6-t)/3, 1/ℯ),N)
            X[:,5,t] = rand(Binomial(20,0.6),N)
            X[:,4,t] = rand(Binomial(20,0.5),N)
        end
    β = Array{Float64}(undef, K, T);
        for t in 1:T
            β[1,t] = 0.75+ 0.25*t
            β[2,t] = log(t)
            β[3,t] = -sqrt(t)
            β[4,t] = ℯ^t - ℯ^(t+1)
            β[5,t] = t
            β[6,t] = t/3
        end
    Y = Array{Float64}(undef, N, T);
        for t in 1:T
            Y[:,t] = X[:,:,t]*β[:,t] + rand(Normal(0, 0.36),N)
        end
end

q2(A,B,C) 


using CSV
function q3()  
    dataq3 = DataFrame(CSV.read("./nlsw88.csv"));
    summarystats=describe(dataq3)
    mean(dataq3[!,"never_married"])
    mean(dataq3[!,"collgrad"])
    prop(freqtable(dataq3, :race))
    freqtable(dataq3,:industry,:occupation)
    datasubset=select(dataq3,"industry","occupation","wage");
    groupDataSubset_industry = groupby(datasubset, :industry);
    combine(groupDataSubset_industry, :wage => mean)
    groupDataSubset_occupation = groupby(datasubset, :occupation)
    combine(groupDataSubset_occupation, :wage => mean) 
end

q3()


function q4()
    load("./firstmatrix.jld")
    function matrixops(A,B)
        # this function takes in matrix A and B and do some matrix operations on them
        if size(A)!= size(B)
            print("inputs must have the same size")
        end
        A.*B,transpose(A)*B, A+B 
    end

    t=matrixops(C,D)
    ttl_exp_array=convert(Array,dataq3.ttl_exp)
    wage_array=convert(Array,dataq3.wage)
    matrixops(ttl_exp_array,wage_array)
end
q4()

