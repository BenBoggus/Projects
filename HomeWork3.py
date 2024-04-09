import numpy as np


def unitVector(magnitude):
    #This function takes the square root of your calculated magnitude
    if np.sqrt(magnitude) == 1:
        return "Vector is a Unit Vector"
    else:
        return "Vector is not a Unit Vector"

def rankSize(rank,size):
    if rank == size:
        print("Matrix is Full Rank\n")
    else: 
        print("Matrix is not full Rank\n")
    return True


##PROBLEM 1
print("Problem 1:")
A = np.mat([[2,6,6,4],[2,3,-2,7],[3,-3,-3,-4],[-8,2,3,7],[5,5,7,5]])
#Part 1
pseudo = np.linalg.pinv(A)
print("Added Column Inverse: \n", pseudo)
print()

#Part 2
X = np.delete(A,1,0)
pseudo2 = np.linalg.pinv(X)
print("Deleted Row Inverse: \n", pseudo2)
print()

#Part 3 and 4
inversion1 = np.matmul(A,pseudo)
answer = "%.4f" % np.linalg.det(inversion1)
print("Determinant AAT: " , answer)
print(f"AAT is Invertible Determinant = {answer} = 0\n")

inversion2 = np.matmul(pseudo,A)
answer2 = "%.4f" % np.linalg.det(inversion2)
print("Determinant ATA:" , answer2)
print(f"ATA is not Invertible Determinant = {answer2} != 0\n")


##PROBLEM 2
print("\nProblem 2:")
#Part 1
u = [0.5,0.4,0.4,0.5,0.1,0.4,0.1] #Putting my vectors into an array
v = [-1,-2,1,-2,3,1,-5]
i = 0 #Initializers
j = 0
magU = 0
magV = 0
while i < len(u): #Looping through my arrays and to calculate the unsquare rooted magnitude
    magU += u[i]**2
    i+=1
while j < len(v):
    magV += v[j]**2
    j+=1
print("\nU: " , unitVector(magU))
print("V: " , unitVector(magV))
print()

#Part 2
print("Dot Product of U.V: ", "%.4f" % np.dot(u,v))

#Part 3
if np.dot(u,v) == 0:
    print("\nVectors are Orthogonal")
else:
    print("\nVectors are not Orthogonal")


##PROBLEM 3
print("\nProblem 3:")
a = np.mat([1,2,5,2,-3,1,2,6,2])
b = [-4,3,-2,2,1,-3,4,1,-2]
c = [3,3,-3,-1,6,-1,2,-5,-7]

#Parts 1 and 2
print("Dot Products are:")
print("v . w = " , np.dot(a,c))
print("v . u = " , np.dot(a,b))
print("u . w = " , np.dot(b,c))
print()

#Part3
print("Solving ||u||2:")
i = 0
magB = 0
while i < len(b):
    magB += b[i]**2
    i += 1
print(np.sqrt(magB))
print()

#Part 4
print("Solving ||w||1:")
j = 0
magC = 0
while j < len(c):
    magC += np.abs(c[j])
    j += 1
print(magC)
print()

##PROBLEM 4
print("\nProblem 4:")
Q = np.mat([[2,4,-2], [5,3,-7]])
R = np.mat([[2,4,-4], [4,4,2], [5,2,3]])
S = np.mat([[2,4,-4], [2,1,8], [2,-1,2]])
P = np.linalg.pinv(Q)
O = np.linalg.pinv(R)

#Part 1
#ATB = ABT when A and B have the same number of Columns
print("\nATB =\n", np.matmul(Q,O))

#Part 2
matrixAdd = S + R
print("\nMatrix Addition of C and B\n" , matrixAdd)

#Part 3
rank1 = np.linalg.matrix_rank(Q)
rank2 = np.linalg.matrix_rank(R)
rank3 = np.linalg.matrix_rank(S)
size1 = Q.shape[1]
size2 = R.shape[1]
size3 = S.shape[1]
print("\nQ:")
rankSize(rank1,size1)
print("R:")
rankSize(rank2,size2)
print("S:")
rankSize(rank3,size3)

#Part 4
frob_norm ="%.4f" % np.linalg.norm(S,'fro')
print("Frobenius Norm of C: ", frob_norm)

#Part 5
l2_NormA = "%.4f" % np.linalg.norm(Q,2)
print("\nEuclidean Norm of A: ", l2_NormA)

#Part 6
inv_B = np.linalg.inv(R)
print("\nInverse of Matrix B:\n" , inv_B)