## 习题

> [!NOTE] **[Luogu 外星密码](https://www.luogu.com.cn/problem/P1928)**
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
#include <bits/stdc++.h>
using namespace std;

// https://leetcode-cn.com/problems/number-of-atoms/

string s;
int n;

string dfs(int & u) {
    string ret;
    while (u < n) {
        if (s[u] == '[') {
            u ++ ;  // '['
            
            int k = u, cnt = 0;
            while (k < n && isdigit(s[k]))
                k ++ ;
            if (k > u) {
                // must be
                cnt = stoi(s.substr(u, k - u));
                u = k;
            }
            
            auto t = dfs(u);
            for (int i = 0; i < cnt; ++ i )
                ret += t;
            
            u ++ ;  // ']'
        } else if (s[u] == ']') {
            break;
        } else {
            int k = u + 1;
            while (k < n && isalpha(s[k]))
                k ++ ;
            
            auto ctx = s.substr(u, k - u);
            u = k;
            
            ret += ctx;    
        }
    }
    return ret;
}

int main() {
    cin >> s;
    n = s.size();
    
    int p = 0;
    cout << dfs(p) << endl;
    
    return 0;
}
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 726. 原子的数量](https://leetcode-cn.com/problems/number-of-atoms/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 经典 递归处理字符串
> 
> 利用 map 保存的技巧, 使用 hash 记录的思路

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using MPSI = map<string, int>;

    MPSI dfs(string& str, int& u) {
        MPSI res;
        while (u < str.size()) {
            if (str[u] == '(') {
                u ++ ;
                auto t = dfs(str, u);
                u ++ ;
                int cnt = 1, k = u;
                while (k < str.size() && isdigit(str[k])) k ++ ;
                if (k > u) {
                    cnt = stoi(str.substr(u, k - u));
                    u = k;
                }
                for (auto& [x, y]: t) res[x] += y * cnt;
            } else if (str[u] == ')') break;
            else {
                int k = u + 1;
                while (k < str.size() && str[k] >= 'a' && str[k] <= 'z') k ++ ;
                auto key = str.substr(u, k - u);
                u = k;
                int cnt = 1;
                while (k < str.size() && isdigit(str[k])) k ++ ;
                if (k > u) {
                    cnt = stoi(str.substr(u, k - u));
                    u = k;
                }
                res[key] += cnt;
            }
        }
        return res;
    }

    string countOfAtoms(string formula) {
        int k = 0;
        auto t = dfs(formula, k);
        string res;
        for (auto& [x, y]: t) {
            res += x;
            if (y > 1) res += to_string(y);
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

> [!NOTE] **[LeetCode 736. Lisp 语法解析](https://leetcode-cn.com/problems/parse-lisp-expression/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 经典 递归处理字符串

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using MPSI = unordered_map<string, int>;

    int get_value(string & str, int & k, MPSI vars) {
        int value;
        if (str[k] == '-' || isdigit(str[k])) {
            int i = k + 1;
            while (isdigit(str[i]))
                i ++ ;
            value = stoi(str.substr(k, i - k));
            k = i;
        } else if (str[k] != '(') {
            string name;
            while (str[k] != ' ' && str[k] != ')')
                name += str[k ++ ];
            value = vars[name];
        } else
            value = dfs(str, k, vars);
        return value;
    }

    int dfs(string & str, int & k, MPSI vars) {
        int value;
        // 跳过 '('
        k ++ ;
        auto type = str.substr(k, 3);
        if (type == "let") {
            // 跳过 "let "
            k += 4;
            while (str[k] != ')') {
                if (str[k] == '(' || str[k] == '-' || isdigit(str[k])) {
                    value = get_value(str, k, vars);
                    break;
                }
                string name;
                while (str[k] != ' ' && str[k] != ')')
                    name += str[k ++ ];
                if (str[k] == ')') {
                    value = vars[name];
                    break;
                }
                // 跳过 ' '
                k ++ ;
                vars[name] = get_value(str, k, vars);
                // 跳过 ' '
                k ++ ;
            }
        } else if (type == "add") {
            k += 4;
            int a = get_value(str, k, vars);
            k ++ ;
            int b = get_value(str, k, vars);
            value = a + b;
        } else {
            k += 5;
            int a = get_value(str, k, vars);
            k ++ ;
            int b = get_value(str, k, vars);
            value = a * b;
        }
        // 跳过 ')'
        k ++ ;
        return value;
    }

    int evaluate(string expression) {
        int k = 0;
        return dfs(expression, k, MPSI());
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

> [!NOTE] **[LeetCode 1096. 花括号展开 II](https://leetcode.cn/problems/brace-expansion-ii/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 定义抽象代数结构

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using SS = set<string>;

    SS add(SS & A, SS & B) {
        SS C(A.begin(), A.end());
        for (auto & x : B)
            if (x.size())
                C.insert(x);
        return C;
    }
    SS mul(SS & A, SS & B) {
        SS C;
        for (auto & x : A)
            for (auto & y : B)
                C.insert(x + y);
        return C;
    }

    string str;
    int k;

    SS dfs() {
        // 维护抽象代数结构: A + B * _
        SS A, B;
        B.insert("");

        while (k < str.size() && str[k] != '}') {
            if (str[k] == ',') {
                k ++ ;
            } else if (str[k] == '{') {
                bool is_add = true;
                if (!k || str[k - 1] != ',')
                    is_add = false;
                
                k ++ ;  // 跳过 {
                auto C = dfs();
                k ++ ;  // 跳过 }

                if (is_add) {
                    A = add(A, B);
                    B = C;
                } else {
                    B = mul(B, C);
                }
            } else {
                bool is_add = true;
                if (!k || str[k - 1] != ',')
                    is_add = false;
                
                string s;
                while (k < str.size() && str[k] >= 'a' && str[k] <= 'z')
                    s += str[k ++ ];
                
                SS C;
                C.insert(s);

                if (is_add) {
                    A = add(A, B);
                    B = C;
                } else {
                    B = mul(B, C);
                }
            }
        }

        return add(A, B);
    }

    vector<string> braceExpansionII(string expression) {
        this->str = expression, this->k = 0;
        auto res = dfs();
        return vector<string>(res.begin(), res.end());
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