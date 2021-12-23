## 习题

### 一般模拟

> [!NOTE] **[AcWing 1364. 序言页码](https://www.acwing.com/problem/content/1366/)**
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

int n;

int main() {
    string name[13] = {
        "M", "CM", "D", "CD", "C", "XC", "L",
        "XL", "X", "IX", "V", "IV", "I"
    };
    int num[13] = {
        1000, 900, 500, 400, 100, 90, 50,
        40, 10, 9, 5, 4, 1
    };
    
    unordered_map<char, int> cnt;
    cin >> n;
    for (int i = 1; i <= n; ++ i )
        for (int j = 0, x = i; j < 13; ++ j )
            while (x >= num[j]) {
                x -= num[j];
                for (auto c : name[j])
                    cnt[c] ++ ;
            }
    
    string cs = "IVXLCDM";
    for (auto c : cs)
        if (cnt[c])
            cout << c << ' ' << cnt[c] << endl;
    
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

> [!NOTE] **[AcWing 1376. 分数化小数](https://www.acwing.com/problem/content/1378/)**
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

const int N = 100010;

int p[N];

int main() {
    int n, d;
    cin >> n >> d;
    
    string res;
    res += to_string(n / d) + '.';
    n %= d;
    
    if (!n) res += '0';
    else {
        memset(p, -1, sizeof p);
        string num;
        // 计算余数出现的位置
        while (n && p[n] == -1) {
            p[n] = num.size();
            n *= 10;
            num += n / d + '0';
            n %= d;
        }
        if (!n) res += num;
        else res += num.substr(0, p[n]) + '(' + num.substr(p[n]) + ')';
    }
    
    for (int i = 0; i < res.size(); ++ i ) {
        cout << res[i];
        if ((i + 1) % 76 == 0) cout << endl;
    }

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

### 丑数



> [!NOTE] **[AcWing 1378. 谦虚数字](https://www.acwing.com/problem/content/1380/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 丑数问题究极板
> 
> 多路归并模型
> 
> 原集合S 用原集合元素生成目标数
> 
> 包含第一个元素的 S1 第k个元素的 Sk

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

struct Data {
    // 数值 原始元素 对应下标
    int v, p, k;
    // 比较符号定义大于 是优先队列默认比较级的原因
    bool operator< (const Data & t) const {
        return v > t.v;
    }
};

int main() {
    int n, k;
    cin >> k >> n;
    n ++ ;
    vector<int> q(1, 1);
    priority_queue<Data> heap;
    
    while (k -- ) {
        int p;
        cin >> p;
        heap.push({p, p, 0});
    }
    
    while (q.size() < n) {
        auto [v, p, k] = heap.top(); heap.pop();
        q.push_back(v);
        heap.push({p * q[k + 1], p, k + 1});
        // 去除重复数
        while (heap.top().v == v) {
            auto [v, p, k] = heap.top(); heap.pop();
            heap.push({p * q[k + 1], p, k + 1});
        }
    }
    cout << q.back() << endl;
    
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

> [!NOTE] **[AcWing 1397. 字母游戏](https://www.acwing.com/problem/content/1399/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> - 分析: 知最多有两个
> 
> - 细节: 键盘映射 筛选

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 5100;

int n;
int cnt[200];
char str[N][8];
char dk[] = "qwertyuiopasdfghjklzxcvbnm";
int dv[] = {
    7, 6, 1, 2, 2, 5, 4, 1, 3, 5,
    2, 1, 4, 6, 5, 5, 7, 6, 3,
    7, 7, 4, 6, 5, 2, 5,
};
int v[200];

int get_score(char s[]) {
    int res = 0;
    for (int i = 0; s[i]; ++ i )
        res += v[s[i]];
    return res;
}

bool check(char a[], char b[]) {
    bool flag = true;
    for (int i = 0; a[i]; ++ i )
        if ( -- cnt[a[i]] < 0)
            flag = false;
    for (int i = 0; b[i]; ++ i )
        if ( -- cnt[b[i]] < 0)
            flag = false;
    for (int i = 0; a[i]; ++ i ) ++ cnt[a[i]];
    for (int i = 0; b[i]; ++ i ) ++ cnt[b[i]];
    return flag;
}

int main() {
    for (int i = 0; i < 26; ++ i ) v[dk[i]] = dv[i];
    
    char s[10];
    cin >> s;
    for (int i = 0; s[i]; ++ i ) cnt[s[i]] ++ ;
    
    while (cin >> str[n], str[n][0] != '.') {
        // 检查能否存下来
        bool flag = true;
        for (int i = 0; str[n][i]; ++ i )
            if ( -- cnt[str[n][i]] < 0)
                flag = false;
        // 加回来
        for (int i = 0; str[n][i]; ++ i )
            cnt[str[n][i]] ++ ;
        if (flag) n ++ ;
    }
    
    int res = 0;
    for (int i = 0; i < n; ++ i ) {
        int score = get_score(str[i]);
        res = max(res, score);
        for (int j = i + 1; j < n; ++ j )
            if (check(str[i], str[j]))
                res = max(res, score + get_score(str[j]));
    }
    
    cout << res << endl;
    for (int i = 0; i < n; ++ i ) {
        int score = get_score(str[i]);
        if (score == res) {
            cout << str[i] << endl;
            continue;
        }
        for (int j = i + 1; j < n; ++ j )
            if (check(str[i], str[j]) && res == score + get_score(str[j]))
                cout << str[i] << ' ' << str[j] << endl;
    }
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


### 思维简化模拟

> [!NOTE] **[AcWing 1367. 派对的灯](https://www.acwing.com/problem/content/1369/)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> 1. 按两次等效于没按
> 
> 2. 按按钮的顺序是无关的
>
> 则 最多可以达到的状态数量只有16种
> 
> 以及 所有按的次数一定小于等于4
>
> 又及：
> 
> - 按 2 + 3 == 1
> - 按 1 + 2 == 3
> - 按 1 + 3 == 2
>
> 次数大于等于3 则必然可以合并其中两个变为小于等于2次的按法
>
> ==> 8种 【无, 1, 2, 3, 4, 14, 24, 34】
>
> 6 因为每六个其状态是一致的

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp

```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

### 数字相关

> [!NOTE] **[SwordOffer 16. 数值的整数次方](https://leetcode-cn.com/problems/shu-zhi-de-zheng-shu-ci-fang-lcof/)**
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
    double myPow(double x, int n) {
          // case
        if (x == 1) return 1;
        else if (x == -1) return n & 1 ? -1 : 1;
        if (n == INT_MIN) return 0;
        int N = n;
        if (n < 0) {
            N = -N;
            x = 1.0 / x;
        }
        double res = 1;
        while (N) {
            if (N & 1) res *= x;
            x *= x;
            N >>= 1;
        }
        return res;
    }
};
```

##### **Python**

```python
# python3
# 快速幂 求 pow(n, k) ===> O(logk)
# 快速幂算法的原理是通过将指数 k 拆分成几个因数相乘的形式，来简化幂运算。
# 原理就是利用位运算里的位移“>>”和按位与“&”运算，代码中 k & 1其实就是取 k 二进制的最低位，用来判断最低位是0还是1，再根据是0还是1决定乘不乘，如果是1，就和当前的 n 相乘，并且 k 要往后移动，把当前的 1 移走，同时 需要 x *= x；

class Solution:
    def myPow(self, x: float, n: int) -> float:
        def fastPow(a, b):
            res = 1
            while b:
                if b & 1:
                    res *= a
                # 注意：b >>= 1 !!!
                b >>= 1
                a *= a
            return res

        if x == 0:return 0
        if n < 0:
            x, n = 1 / x, -n
        return fastPow(x, n)
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[SwordOffer 20. 表示数值的字符串](https://leetcode-cn.com/problems/biao-shi-shu-zhi-de-zi-fu-chuan-lcof/)**
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
// 标准写法
class Solution {
public:
    int n;

    bool scanUnsignedInt(string & s, int & i) {
        int p = i;
        while (i < n && isdigit(s[i]))
            i ++ ;
        return i > p;
    }

    bool scanInt(string & s, int & i) {
        if (i < n && (s[i] == '+' || s[i] == '-'))
            i ++ ;
        return scanUnsignedInt(s, i);
    }

    bool isNumber(string s) {
        this->n = s.size();
        int i = 0;

        while (i < n && s[i] == ' ')
            i ++ ;
        
        bool flag = scanInt(s, i);
        if (i < n && s[i] == '.')
            flag = scanUnsignedInt(s, ++ i ) || flag;
        if (i < n && (s[i] == 'e' || s[i] == 'E'))
            flag = scanInt(s, ++ i ) && flag;
        
        while (i < n && s[i] == ' ')
            i ++ ;

        return flag && i == n;
    }
};
```

##### **Python**

```python
# python3
# (模拟) O(n)
# 这道题边界情况很多，首先要考虑清楚的是有效的数字格式是什么，这里用A[.[B]][e|EC]或者.B[e|EC]表示，其中A和C都是整数(可以有正负号也可以没有)，B是无符号整数。

# 那么我们可以用两个辅助函数来检查整数和无符号整数的情况，从头到尾扫一遍字符串然后分情况判断，注意这里要传数字的引用或者用全局变量。


class Solution:
    def isNumber(self, s: str) -> bool:
        n = len(s)
        i = 0

        # 用来判断是否存在正数
        def scanUnsignedInt():
            # 用 nonlocal 踩坑，不能把 i 放进函数里传递
            nonlocal i  
            p = i 
            while i < n and s[i].isdigit():
                i += 1
            return p < i
        
        # 用来判断是否存在整数
        def scanInt():
            nonlocal i
            if i < n and (s[i] == '+' or s[i] == '-'):
                i += 1
            return scanUnsignedInt()

        while i < n and s[i] == ' ':
            i += 1

        flag = scanInt()
        if i < n and s[i] == '.':
            i += 1
            flag = scanUnsignedInt() or flag
        if i < n and (s[i] == 'e' or s[i] == 'E'):
            i += 1
            flag = scanInt() and flag

        while i < n and s[i] == ' ':
            i += 1
        return flag and i == n
```

<!-- tabs:end -->
</details>

<br>

* * *