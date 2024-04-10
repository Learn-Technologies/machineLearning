import numpy as np

n=int(input('enter years:'))

x=np.array([2,3,5,13,8,16,11,1,9])
y=np.array([15,28,42,64,50,90,58,8,54])

xmean=x.mean()
ymean=y.mean()

lx=[]
for i in x:
    lx.append(i-xmean)
    
##print(lx)

ly=[]
for i in y:
    ly.append(i-ymean)
##print(ly)

l=[]
s=0
sx=0
for i in range(len(lx)):
    l.append(lx[i]*ly[i])
    s+=lx[i]*ly[i]
    sx+=lx[i]*lx[i]
##print(l)
##print(s)
##print(sx)
a=s/sx
b=ymean-(a*xmean)
##print(a)
##print(b)
##
##print(xmean)
##print(ymean)

print('salary is:',b+(a*n))






