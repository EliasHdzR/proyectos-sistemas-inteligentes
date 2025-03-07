my_dict = {}
grades = {'Ana':'B'}

print(grades['Ana'])
#print(grades['Sylvan'])

grades['Marco'] = 'X'
print(grades)
print(type(grades))
print(len(grades))

for i in grades.keys():
    print (i)

for i in grades.values():
    print (i)

for i in grades:
    print (i, grades[i])

exit()

L=[0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9]
print(sorted(L))
print(L)
print(L.sort())
print(L)
print(L.reverse)
print(L)

exit()

A=1244
A = str(A)
print(list(A))

s = "Marco Aurelio NUño"
print(s.split(" "))
exit()

L=[0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9]
L.remove(0)
print(L)
exit()

#L=["Marco","Aurelio"]
#L.append([1,2,3])
#L.extend([1,2,3])
#L.extend([1,2])

print(L)

exit()

sum = ""
for i in L:
    sum+=i
print (sum)

exit()

def quotient_and_remainder(x, y):
    q = x // y # // es una division que devuelve un entero
    r = x % y
    return (q, r)
(quot, rem) = quotient_and_remainder(6,1)
print(quot,rem)

exit()

def f(y):
    global x
    x = 1
    x += 1
    print(x)

x = 5
f(x)
print(x)

exit()

def func_a(i):
    print ('inside func_a')
    return "ABC"
def func_b(y):
    print ('inside func_b')
    return y
def func_c(z):
    print ('inside func_c')
    return z(1)

print(func_a(1))
print(5 + func_b(2))
print (func_c(func_a))

exit()

def is_even( i ):
    """
    Input: i, a positive int
    Returns True if i is even, otherwise False
    """
    i=i+1
    print("inside is_even")
    return i%2 == 0

nombre="marco"
print(len(nombre))
print(is_even(3))
print(is_even(4))

exit()

for n in range(10,0,-1):
    print(n)
    if(n<5):
        break
exit()

n = input("You're in the Lost Forest. Go left or right? ")
valor = 10
#while n == "right":
while valor != 0:
    print(valor)
    valor -= 1
    #n = input("You're in the Lost Forest. Go left or right? ")
print("You got out of the Lost Forest!")

exit()

x = float(input("Enter a number for x: "))
y = float(input("Enter a number for y: "))
if x == y:
    print("x and y are equal")
    if y != 0:
        print("therefore, x / y is", x/y)

elif x < y:
    print("x is smaller")
else:
    print("y is smaller")
print("thanks!")

exit()

texto1 = int(input("Escriba el OP1: "))
texto2 = int(input("Escriba el OP2: "))
print(texto1 < texto2)
print(not(texto1 == texto2))
print(texto1 > texto2)

exit()

nombre = "Marco"
apellido = "Aurelio"
valor = 3
boleana = False
print(nombre+apellido,valor)
print(nombre+apellido+str(valor))
print(nombre*valor)
print(nombre+str(boleana))
print(nombre,int(boleana))
print(nombre, boleana)

exit()

a=3
c=3.0
b=float(a)
d=int(c)

r1=a+c
r2=a-c
r3=a*c
r4=a/a
print(r1,r2,r3,r4)

#print (type(a),type(b),type(c),type(d))
#print (a,b,c,d)
