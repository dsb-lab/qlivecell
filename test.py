from collections import deque
q = deque([],5)
q.pop()
for i in range(10):
    q.append(i)
    print(q)

a = q.pop()