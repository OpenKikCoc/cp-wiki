## 习题

### 栈与队列

> [!NOTE] **[SwordOffer 09. 用两个栈实现队列](https://leetcode-cn.com/problems/yong-liang-ge-zhan-shi-xian-dui-lie-lcof/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class MyQueue {
public:
    stack<int> in, out;

    /** Initialize your data structure here. */
    MyQueue() {
    }
    
    /** Push element x to the back of queue. */
    void push(int x) {
        in.push(x);
    }
    
    /** Removes the element from in front of queue and returns that element. */
    int pop() {
        if (out.empty()) {
            while (in.size()) {
                out.push(in.top());
                in.pop();
            }
        }
        int t = out.top();
        out.pop();
        return t;
    }
    
    /** Get the front element. */
    int peek() {
        if (out.empty()) {
            while (in.size()) {
                out.push(in.top());
                in.pop();
            }
        }
        return out.top();
    }
    
    /** Returns whether the queue is empty. */
    bool empty() {
        return in.empty() && out.empty();
    }
};

/**
 * Your MyQueue object will be instantiated and called as such:
 * MyQueue obj = MyQueue();
 * obj.push(x);
 * int param_2 = obj.pop();
 * int param_3 = obj.peek();
 * bool param_4 = obj.empty();
 */
```

##### **Python**

```python
# 算法流程：1. self.A 是用来压入数据的，self.B 用来弹出数据
# 2. 在append 的时候，就直接压入 self.A 中
# 3. 在delete 队头元素时，首先先判断 self.B 中是否有元素，有的话，直接弹出就可以。如果没有的话，再去看self.A 中是否有元素，如果 self.A 中也没有元素的话，说明当前模拟的队列已经是空了，直接return -1；
# 4. 当self.A 还有元素时，把self.A 的元素全部 pop 到 self.B 中，返回 self.B 的top 元素即可

class CQueue:
    def __init__(self):
        self.A = []
        self.B = []

    def appendTail(self, value: int) -> None:
        self.A.append(value)

    def deleteHead(self) -> int:
        #先判断self.B是否有元素，有的话 直接弹出
        if self.B:return self.B.pop() 
        # B中没有元素，那就看A是不是有新加入的元素，没有的话 返回-1.
        if not self.A:return -1 
        # 有的话，再把A中元素全部压入到B中
        while self.A: 
            self.B.append(self.A.pop())
        return self.B.pop()
```

##### **Python 2**

```python
# 需要一个辅助栈；
# A栈负责入栈所有元素，辅助栈B就负责pop和peek
class MyQueue(object):

    def __init__(self):
        self.A = []
        self.B = []

    def push(self, x)：
        self.A.append(x)
        

    def pop(self):
        if self.B:return self.B.pop()
        while self.A:
            self.B.append(self.A.pop())
        return self.B.pop()

    def peek(self):
        if self.B:return self.B[-1]
        while self.A:
            self.B.append(self.A.pop())
        return self.B[-1]

    def empty(self):
        return not self.A and not self.B
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 225. 用队列实现栈](https://leetcode-cn.com/problems/implement-stack-using-queues/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class MyStack {
public:
    /** Initialize your data structure here. */

    queue<int> q, w;
    MyStack() {

    }

    /** Push element x onto stack. */
    void push(int x) {
        q.push(x);
    }

    /** Removes the element on top of the stack and returns that element. */
    int pop() {
        while (q.size() > 1) w.push(q.front()), q.pop();
        int t = q.front();
        q.pop();
        while (w.size()) q.push(w.front()), w.pop();
        return t;
    }

    /** Get the top element. */
    int top() {
        while (q.size() > 1) w.push(q.front()), q.pop();
        int t = q.front();
        q.pop();
        while (w.size()) q.push(w.front()), w.pop();
        q.push(t);
        return t;
    }

    /** Returns whether the stack is empty. */
    bool empty() {
        return q.empty();
    }
};

/**
 * Your MyStack object will be instantiated and called as such:
 * MyStack* obj = new MyStack();
 * obj->push(x);
 * int param_2 = obj->pop();
 * int param_3 = obj->top();
 * bool param_4 = obj->empty();
 */
```

##### **Python**

```python
class MyStack:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.q=[]


    def push(self, x: int) -> None:
        """
        Push element x onto stack.
        """
        self.q.append(x)
        n=len(self.q)
        while n>1:
            self.q.append(self.q.pop(0))
            n-=1



    def pop(self) -> int:
        """
        Removes the element on top of the stack and returns that element.
        """
        return self.q.pop(0)


    def top(self) -> int:
        """
        Get the top element.
        """
        return self.q[0]


    def empty(self) -> bool:
        """
        Returns whether the stack is empty.
        """
        return not bool(self.q)



# Your MyStack object will be instantiated and called as such:
# obj = MyStack()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.top()
# param_4 = obj.empty()
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 232. 用栈实现队列](https://leetcode-cn.com/problems/implement-queue-using-stacks/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class MyQueue {
    stack<int> in, out;
public:
    /** Initialize your data structure here. */
    MyQueue() {

    }
    
    /** Push element x to the back of queue. */
    void push(int x) {
        in.push(x);
    }
    
    /** Removes the element from in front of queue and returns that element. */
    int pop() {
        if(out.empty()) {
            int tot = in.size(), t;
            while(tot--) {
                t = in.top();
                out.push(t);
                in.pop();
            }
        }
        int t = out.top();
        out.pop();
        return t;
    }
    
    /** Get the front element. */
    int peek() {
        if(out.empty()) {
            int tot = in.size(), t;
            while(tot--) {
                t = in.top();
                out.push(t);
                in.pop();
            }
        }
        int t = out.top();
        //out.pop();
        return t;
    }
    
    /** Returns whether the queue is empty. */
    bool empty() {
        return in.empty() && out.empty();
    }
};

/**
 * Your MyQueue object will be instantiated and called as such:
 * MyQueue* obj = new MyQueue();
 * obj->push(x);
 * int param_2 = obj->pop();
 * int param_3 = obj->peek();
 * bool param_4 = obj->empty();
 */
```

##### **Python**

```python
class MyQueue:

    def __init__(self):
        self.A = []
        self.B = []


    def push(self, x: int) -> None:
        while self.B:
            self.A.append(self.B.pop())
        self.A.append(x)


    def pop(self) -> int:
        while self.A:
            self.B.append(self.A.pop())
        return self.B.pop()


    def peek(self) -> int:
        while self.A:
            self.B.append(self.A.pop())
        return self.B[-1]


    def empty(self) -> bool:
        return not self.A and not self.B
```

<!-- tabs:end -->
</details>

<br>

* * *

### 栈

> [!NOTE] **[SwordOffer 30. 包含min函数的栈](https://leetcode-cn.com/problems/bao-han-minhan-shu-de-zhan-lcof/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class MinStack {
public:
    stack<int> st, mst;

    /** initialize your data structure here. */
    MinStack() {

    }
    
    void push(int x) {
        st.push(x);
        if (mst.empty() || mst.top() >= x)
            mst.push(x);
    }
    
    void pop() {
        if (st.top() == mst.top())
            mst.pop();
        st.pop();
    }
    
    int top() {
        return st.top();
    }
    
    int getMin() {
        return mst.top();
    }
};
```

##### **Python**

```python
# python3
# 借助一个辅助栈min-B 专门存储最小元素，栈顶存储的就是当前栈的最小元素。
# 压入栈的时候，需要判断B中的情况，如果B不存在 or B中栈顶元素大于当前元素，那么应该把当前元素压入栈中，否则就不压入到B中【这是由于栈具有先进后出性质，所以在当前元素被弹出之前，栈中一直存在一个数比该数小，所以当前元素一定不会被当作最小数输出】
# 在pop的时候， 需要判断A pop出去的元素是否等于min-B的栈顶元素，如果是的话，就需要把B 的栈顶元素弹出。

class MinStack(object):
    def __init__(self):
        self.A,self.B = [], []

    def push(self, x):
        self.A.append(x)
        if not self.B or self.B[-1] >= x:
            self.B.append(x)

    def pop(self) -> None:
        if not self.A:
            return -1
        x = self.A[-1]
        if x == self.B[-1]:
            self.B.pop()
        return self.A.pop()

    def top(self):
        if self.A:
            return self.A[-1]

    def getMin(self):
        return self.B[-1]
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[SwordOffer 31. 栈的压入、弹出序列](https://leetcode-cn.com/problems/zhan-de-ya-ru-dan-chu-xu-lie-lcof/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    bool validateStackSequences(vector<int>& pushed, vector<int>& popped) {
        int m = pushed.size(), n = popped.size();
        if (m != n) return false;
        int p = 0;
        stack<int> s;
        for (int i = 0; i < n; ++ i ) {
            s.push(pushed[i]);
            while (!s.empty() && s.top() == popped[p]) {
                s.pop();
                 ++ p;
            }
        }
        return p == n;
    }
};
```

##### **Python**

```python
# python3 

# 思路：相当于在比较两个字符串是否匹配；但是 对于A而言，当前字符不一定能和B马上匹配使用，可能后续才会用到，所以需要用到栈的结构
# 用一个辅助栈s，将入栈序列压入s, 压入后 栈顶元素 和 出栈序列做对比，如果相同 就把用过的元素都pop出去（对于s 是pop出去，对于出栈序列 是指针++1）

class Solution:
    def validateStackSequences(self, A: List[int], B: List[int]) -> bool:
        n, m = len(A), len(B)
        # 踩坑：特殊case
        if n != m:return False    
        # 踩坑：特殊case
        if not A and not B:return True   
        stack, k = [], 0
        for i in range(n):
            # 踩坑，需要先压入再判断，否则会存在最后一个元素在栈内 没有被比较pop出去
            stack.append(A[i])  
            while stack and stack[-1] == B[k]:
                stack.pop()
                k += 1
        return k == m
```

<!-- tabs:end -->
</details>

<br>

* * *

### 堆与队列

> [!NOTE] **[AcWing 字符流中第一个只出现一次的字符]()**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// AcWing
class Solution{
public:
    unordered_map<char, int> hash;
    queue<char> q;

    //Insert one char from stringstream
    void insert(char ch){
        q.push(ch);
        if ( ++ hash[ch] > 1)
            while (q.size() && hash[q.front()] > 1)
                q.pop();
    }
    //return the first appearence once char in current stringstream
    char firstAppearingOnce(){
        return q.empty() ? '#' : q.front();
    }
};
```

##### **Python**

```python
# 如何把O(n*n)算法优化成O(N) : 双指针，单调队列（找工作面试时的优化方式）
# 首先看一下答案是不是单调的？从前往后走，找第一个没有被划掉的位置。（答案的位置是从前往后递增的，具备一定的单调性）
# 可以用双指针算法 进行优化。（具备单调性，才能用双指针算法）
# 在实现的时候，可以换一种实现方式，比如队列（队列本身就是个双指针）
# 判断队头元素出现的次数是否大于1，如果大于1了，就直接删除。

from collections import deque

class Solution:  
    def __init__(self):
        self.dic = {}
        self.q = deque()
        
    def firstAppearingOnce(self):
        if not self.q:return '#'
        return self.q[0]
        
    def insert(self, char):
        if char in self.dic:
            self.dic[char] += 1
            while self.q and self.dic[self.q[0]] > 1:
                self.q.popleft()
        else:
            self.dic[char] = 1
            self.q.append(char)
```

<!-- tabs:end -->
</details>

<br>

* * *
