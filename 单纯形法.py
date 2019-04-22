from numpy import *
import numpy as np

def solve_stdLP(A,C,b):
	control=True
	m,n=A.shape
	a=np.column_stack([A,np.mat(eye(m))])
	c=np.concatenate((np.mat(zeros(C.shape)),np.mat(ones((m,1)))))
	index_list_b=[i for i in range(n+1,n+m+1,1)]
	while control:
		judge,indices=standard_lp_solve(a,c,b,index_list_b)
		if judge>1e-6:
			control=False
			print('Infeasible Solution')
		elif judge<1e-6 and max(indices)+1<n+1:
			control=False
			standard_lp_solve(A,C,b,[index+1 for index in indices],switch=True)
		else:
			temp_b=a[:,indices].copy()
			c_b=np.mat(zeros((len(indices),1)))
			c_b=c[indices,:]
			eta=c_b.T*temp_b.I*a-c.T
			temp_index=np.argmax(eta[0,:])+1
			indices,count=find_next(a,temp_b,b,temp_index,indices)
			delete(a,count,axis=1)


def standard_lp_solve(A,C,b,index_list_B=False,switch=False):
	#index_list_B is a list which contains the index of feasible base
	#A is a matrix,and C is the cost matrix(a column vector)
	m,n=A.shape
	x=np.mat(zeros(C.shape)).T
	while True:
		indices=[]
		for index in index_list_B:
			indices.append(index-1)
		B=A[:,indices].copy()
		C_B=np.mat(zeros((len(indices),1)))
		C_B=C[indices,:]
		#eta is a row vector
		eta=C_B.T*B.I*A-C.T
		temp_index=np.argmax(eta[0,:])+1
		if eta.max()<=1e-6:
			for i in range(m):
				x[:,indices[i]]=(B.I*b.T)[i]
			z=x*C
			if switch:
				for index in indices:
					print('x'+str(int(index+1))+'='+str(float(x[:,index])))
				print('else=0')
				print('z='+str(float(z)))
			return z,indices
		elif (B.I*(A[:,temp_index-1])).max()<=1e-6:
			print('Unbounded')
			return 0
		else:
			index_list_B,count=find_next(A,B,b,temp_index,index_list_B)

def find_next(A,B,b,temp_index,index_list_B):
	temp1=(B.I*(A[:,temp_index-1]))
	temp2=B.I*b.T
	temp4=inf
	temp3=[]
	for i in range(len(temp1)):
		if temp1[i] >0 and temp4>float(temp2[i]/temp1[i]):
			temp4=float(temp2[i]/temp1[i])
			count=i
	index_list_B[count]=temp_index
	return index_list_B,count
		

def Solve():
	b=np.mat([24,8])
	A=np.mat([[2,4,10,-1,0],[5,1,5,0,-1]])
	C=np.mat([4,2,6,0,0]).T
	solve_stdLP(A,C,b)

Solve()
	
	
