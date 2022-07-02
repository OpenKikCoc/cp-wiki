## 栈

![](./images/stack.svg)

栈是 OI 中常用的一种线性数据结构，请注意，本文主要讲的是栈这种数据结构，而非程序运行时的系统栈/栈空间

栈的修改是按照后进先出的原则进行的，因此栈通常被称为是后进先出（last in first out）表，简称 LIFO 表。

> [!WARNING]
> 
> LIFO 表达的是 **当前在容器** 内最后进来的最先出去。
> 
> 我们考虑这样一个栈

```text
    push(1)
    pop(1)
    push(2)
    pop(2)
```

    如果从整体考虑，1 最先入栈，最先出栈，2 最后入栈，最后出栈，这样就成了一个先进先出表，显然是错误的。
    
    所以，在考虑数据结构是 LIFO 还是 FIFO 的时候，应当考虑在当前容器内的情况。

我们可以方便的使用数组来模拟一个栈，如下：

> [!NOTE] **[AcWing 828. 模拟栈](https://www.acwing.com/problem/content/830/)**
> 
> 题意: TODO

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;
const int maxn = 1e5 + 5;

int st[maxn], tot;

void init() { tot = 0; }

void push(int x) { st[++tot] = x; }

void pop() {
    if (tot) --tot;  // 保险
}

bool empty() { return !tot; }

int query() { return st[tot]; }

int main() {
    init();
    int m, x;
    string c;
    cin >> m;
    while (m--) {
        cin >> c;
        if (c == "push") {
            cin >> x;
            push(x);
        } else if (c == "pop") {
            pop();
        } else if (c == "empty") {
            if (empty())
                cout << "YES" << endl;
            else
                cout << "NO" << endl;
        } else if (c == "query") {
            cout << query() << endl;
        }
    }
}
```

##### **Python**

```python
if __name__ == '__main__':
    N = 100010
    # 初始化栈 和 栈顶的下标；注意 是栈顶的下标！！！
    stk = [0] * N
    tt = 0

    n = int(input())

    for i in range(n):
        op = input().split()
        # 插入操作
        if op[0] == 'push':
            x = int(op[1])
            tt += 1
            stk[tt] = x
        # pop操作
        elif op[0] == 'pop':
            tt -= 1
        # 查询是否为空
        elif op[0] == 'empty':
            if tt:
                print('NO')
            else:
                print('YES')
        # 取出栈顶元素
        else:
            print(stk[tt])

```

<!-- tabs:end -->
</details>

<br>

* * *


同时 STL 也提供了一个方法 `std::stack`

```cpp
#include <stack>
// stack 构造 ：
1. stack<Typename T> s;
2. stack<Typename T, Container> s;
/* stack 的 Container 需要满足有如下接口 ：
 * back()
 * push_back()
 * pop_back()
 * 标准容器 std::vector / deque / list 满足这些要求
 * 如使用 1 方式构造，默认容器使用 deque
 */
```

`std::stack` 支持赋值运算符 `=`

元素访问：

`s.top()` 返回栈顶

容量：

`s.empty()` 返回是否为空

`s.size()` 返回元素数量

修改：

`s.push()` 插入传入的参数到栈顶

`s.pop()` 弹出栈顶

其他运算符：

`==`、`!=`、`<`、`<=`、`>`、`>=` 可以按照字典序比较两个 `stack` 的值

在 Python 中，你可以使用列表来模拟一个队列

```python
stk = [5, 1, 4]

# 使用 append() 向栈顶添加元素
stk.append(2)
stk.append(3)
# >>> stk
# [5, 1, 4, 2, 3]

# 使用 pop 取出栈顶元素
stk.pop()
# >>> stk
# [5, 1, 4, 2]

# 使用 clear 清空栈
stk.clear()
```

## 习题

### 弹出匹配

> [!NOTE] **[LeetCode 20. 有效的括号](https://leetcode-cn.com/problems/valid-parentheses/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 1**

```cpp
class Solution {
public:
    bool match(char a, char b) {
        if (a == '(') return b == ')';
        else if (a == '[') return b == ']';
        else if (a == '{') return b == '}';
        return false;
    }
    bool isValid(string s) {
        stack<char> st;
        for (int i = 0; i < s.size(); ++ i ) {
            char c = s[i];
            if (!st.empty() && match(st.top(), c)) st.pop();
            else st.push(c);
        }
        return st.empty();
    }
};
```

##### **C++ 2**

```cpp
class Solution {
public:
    bool isValid(string s) {
        stack<char> stk;

        for (auto c : s) {
            if (c == '(' || c == '[' || c == '{') stk.push(c);
            else {
                if (stk.size() && abs(stk.top() - c) <= 2) stk.pop();
                else return false;
            }
        }

        return stk.empty();
    }
};
```

##### **Python**

```python
# 把左括号 都加入到stack中，当来了右括号时，右括号的对应值 与 栈顶元素对比：如果相同 就继续对比
# 如果不同，就直接return False 
# 如果栈为空，但此时来了一个 左括号，那可以直接返回 False

class Solution:
    def isValid(self, s: str) -> bool:
        my_dict = {")":"(", "}":"{", "]":"["}
        stack = []
        for c in s:
            if c not in my_dict:   
                stack.append(c)
            # ！！踩坑：需要先判断stack不为空；并且逻辑符号是 or （就是如果栈为空了，还有右括号来的话，那就可以直接return False了）
            elif not stack or my_dict[c] != stack.pop(): 
                return False 
         # 踩坑！最后要判断栈是否为空，栈为空的时候 才能返回True 存在可能还有左扩号没有被匹配完的情况
        return not stack 
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 591. 标签验证器](https://leetcode-cn.com/problems/tag-validator/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 栈 经典线性处理字符串

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    bool isValid(string code) {
        stack<string> stk;
        for (int i = 0; i < code.size(); i ++ ) {
            if (code[i] == '<' && code.substr(i, 9) != "<![CDATA[") {  // 标签
                int j = i + 1;
                string tag_name;
                bool is_end = false;
                while (j < code.size() && code[j] != '>') {
                    char c = code[j ++ ];
                    if (c == '/' && j == i + 2) {  // 结束标签
                        is_end = true;
                        continue;
                    }
                    if (c < 'A' || c > 'Z') return false;  // 标签中有非大写英文字母
                    tag_name += c;
                }
                if (j == code.size()) return false;  // 标签没有以 > 结束
                if (tag_name.size() < 1 || tag_name.size() > 9) return false;  // 标签名长度不在1~9之间

                if (is_end) {
                    if (stk.empty() || stk.top() != tag_name) return false;  // 结束标签没有匹配
                    stk.pop();
                } else {
                    if (i && stk.empty()) return false;  // 至少有两组标签
                    stk.push(tag_name);
                }
                i = j;
            } else {  // 处理tag_content
                if (stk.empty()) return false;  // tag_content 不在标签内
                if (code.substr(i, 9) == "<![CDATA[") {  // 处理CDATA
                    int j = i;
                    while (j < code.size() && code.substr(j, 3) != "]]>") j ++ ;
                    if (j == code.size()) return false;  // CDATA没有结尾
                    i = j + 2;
                }
            }
        }

        return stk.empty();
    }
};
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 1249. 移除无效的括号](https://leetcode-cn.com/problems/minimum-remove-to-make-valid-parentheses/)**
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
    string minRemoveToMakeValid(string s) {
        int n = s.size();
        stack<int> st;
        vector<bool> vis(n + 1);
        for (int i = 0; i < n; ++i) {
            if (s[i] == '(')
                st.push(i);
            else if (s[i] == ')') {
                if (!st.empty()) {
                    int t = st.top();
                    st.pop();
                    vis[t] = vis[i] = true;  // 合法
                }
            } else
                vis[i] = true;
        }
        string res;
        for (int i = 0; i < n; ++i)
            if (vis[i]) res.push_back(s[i]);
        return res;
    }
};
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 1541. 平衡括号字符串的最少插入次数](https://leetcode-cn.com/problems/minimum-insertions-to-balance-a-parentheses-string/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
>
> 某个 ( 需在连续两个的 )) 左侧 才可以形成平衡。
>
> 模拟一遍即可，注意模拟规则。
>
> 有用栈模拟的代码：
>
> >   在检查字符的时候分两种情况：
> >
> >   1.字符是 '(' 这个时候只需要把它的位置压进栈里就行
> >
> >   2.字符是 ')' 这个时候，检查它后面的那个字符是不是')'，如果不是')'，说明需要插入一个')'。然后再检查栈是不是空的，如果非空则匹配pop掉；如果栈是空的，说明没有与之匹配的'('，所以需要插入一个'('。直到检查结束，统计到的需要插入的数目。
> >
> >   字符串遍历完毕 当字符串遍历完毕的时候，如果这时候栈不空，那么就把栈里的左括号数目乘以2加到结果上，因为每个左括号需要插入两个右括号去匹配。
> >
> >   栈的代替 由于栈里面只有'('，因此我们可以用一个整形数据代替栈里面左括号的数目，这样可以节省时间。
>
> 思路与之前的仍然是一致的，用数值 `l` 代替了栈

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int minInsertions(string s) {
        int n = s.size(), l = 0, res = 0;
        for (int i = 0; i < n; ++i) {
            if (s[i] == '(')
                ++l;
            else {
                if (i + 1 < n && s[i + 1] == ')')
                    ++i;    // 匹配
                else
                    ++res;  // 缺少一个 加一个来匹配
                if (l > 0)
                    --l;    // 匹配
                else
                    ++res;  // 缺少 (
            }
        }
        res += 2 * l;
        return res;
    }
};
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

### 对顶栈


### 链表

> [!NOTE] **[LeetCode 2296. 设计一个文本编辑器](https://leetcode.cn/problems/design-a-text-editor/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> - 可以链表模拟，重点在于 STL 的一些返回值及操作。详见 STL 部分
> 
> - 标准做法 **对顶栈**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ STL**

```cpp
class TextEditor {
public:
    list<char> L;
    list<char>::iterator p;
    
    TextEditor() {
        p = L.begin();
    }
    
    string print() {
        string t;
        for (auto c : L)
            t.push_back(c);
        return t;
    }
    
    void addText(string text) {
        for (auto c : text)
            L.insert(p, c); // ATTENTION STL insert 后的变化
    }
    
    int deleteText(int k) {
        int c = 0;
        for (; k && p != L.begin(); -- k )
            p = L.erase(prev(p)), c ++ ;    // ATTENTION erase 的返回值
        return c;
    }
    
    string getText() {
        string t;
        auto it = p;
        for (int k = 10; k && it != L.begin(); -- k ) {
            it = prev(it);
            t.push_back(*it);
        }
        reverse(t.begin(), t.end());
        return t;
    }
    
    string cursorLeft(int k) {
        for (; k && p != L.begin(); -- k )
            p = prev(p);
        return getText();
    }
    
    string cursorRight(int k) {
        for (; k && p != L.end(); -- k )
            p = next(p);
        return getText();
    }
};

/**
 * Your TextEditor object will be instantiated and called as such:
 * TextEditor* obj = new TextEditor();
 * obj->addText(text);
 * int param_2 = obj->deleteText(k);
 * string param_3 = obj->cursorLeft(k);
 * string param_4 = obj->cursorRight(k);
 */
```

##### **C++ 对顶栈**

```cpp
class TextEditor {
public:
    vector<char> l, r;
    
    TextEditor() {
        l.clear(), r.clear();
    }
    
    void addText(string text) {
        for (auto c : text)
            l.push_back(c);
    }
    
    int deleteText(int k) {
        int c = 0;
        while (l.size() && k)
            l.pop_back(), k -- , c ++ ;
        return c;
    }
    
    string getText() {
        string t;
        int n = l.size();
        for (int i = max(0, n - 10); i < n; ++ i )
            t.push_back(l[i]);
        return t;
    }
    
    string cursorLeft(int k) {
        while (l.size() && k) {
            char c = l.back();
            r.push_back(c), l.pop_back();
            k -- ;
        }
        return getText();
    }
    
    string cursorRight(int k) {
        while (r.size() && k) {
            char c = r.back();
            l.push_back(c), r.pop_back();
            k -- ;
        }
        return getText();
    }
};

/**
 * Your TextEditor object will be instantiated and called as such:
 * TextEditor* obj = new TextEditor();
 * obj->addText(text);
 * int param_2 = obj->deleteText(k);
 * string param_3 = obj->cursorLeft(k);
 * string param_4 = obj->cursorRight(k);
 */
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

### 栈维护

> [!NOTE] **[LeetCode 636. 函数的独占时间](https://leetcode-cn.com/problems/exclusive-time-of-functions/)**
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
    // 是时间 不是时刻
    vector<int> exclusiveTime(int n, vector<string>& logs) {
        vector<int> res(n);
        stack<int> st;
        int last;
        for (auto & log : logs) {
            int x = log.find(':'), y = log.substr(x + 1).find(':') + x + 1;
            int id = stoi(log.substr(0, x)), ts = stoi(log.substr(y + 1));
            if (log[x + 1] == 's') {
                if (st.size()) res[st.top()] += ts - last;
                st.push(id);
                last = ts;
            } else {
                // 对于同一个数 开始的时候是作为后一段 结束的时候作为前一段
                res[st.top()] += ts - last + 1;
                st.pop();
                last = ts + 1;
            }
        }
        return res;
    }
};
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

### 其他操作

> [!NOTE] **[LeetCode 1190. 反转每对括号间的子串](https://leetcode-cn.com/problems/reverse-substrings-between-each-pair-of-parentheses/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 栈操作

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 1**

```cpp
class Solution {
public:
    string reverseParentheses(string s) {
        vector<string> v(1);
        for (auto c : s) {
            if (c == '(') v.push_back("");
            else if (c == ')') {
                auto t = v.back(); v.pop_back();
                reverse(t.begin(), t.end());
                v.back() += t;
            } else v.back() += c;
        }
        return v[0];
    }
};
```

##### **C++ 2**

```cpp
class Solution {
public:
    string reverseParentheses(string s) {
        stack<string> st;
        st.push("");
        string res;
        for (auto c : s) {
            if (c == '(') st.push("");
            else if (c == ')') {
                auto t = st.top(); st.pop();
                reverse(t.begin(), t.end());
                st.top() += t;
            } else st.top().push_back(c);
        }
        return st.top();
    }
};
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

### trick

> [!NOTE] **[LeetCode 2197. 替换数组中的非互质数](https://leetcode-cn.com/problems/replace-non-coprime-numbers-in-array/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 思维 栈

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    vector<int> replaceNonCoprimes(vector<int>& nums) {
        vector<int> res;
        for (auto x : nums) {
            while (res.size() > 0 && __gcd(res.back(), x) > 1) {
                x = x / __gcd(res.back(), x) * res.back();
                res.pop_back();
            }
            res.push_back(x);
        }
        return res;
    }
};
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *