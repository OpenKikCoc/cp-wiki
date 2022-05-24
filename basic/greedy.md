算法（英语：greedy algorithm），是用计算机来模拟一个“”的人做出决策的过程。这个人十分贪婪，每一步行动总是按某种指标选取最优的操作。而且他目光短浅，总是只看眼前，并不考虑以后可能造成的影响。

## 证明方法

算法有两种证明方法：反证法和归纳法。一般情况下，一道题只会用到其中的一种方法来证明。

1. 反证法：如果交换方案中任意两个元素/相邻的两个元素后，答案不会变得更好，那么可以推定目前的解已经是最优解了。
2. 归纳法：先算得出边界情况（例如 $n = 1$）的最优解 $F_1$，然后再证明：对于每个 $n$，$F_{n+1}$ 都可以由 $F_{n}$ 推导出结果。

## 要点

### 常见题型

在提高组难度以下的题目中，最常见的有两种。

- 「我们将 XXX 按照某某顺序排序，然后按某种顺序（例如从小到大）选择。」。
- 「我们每次都取 XXX 中最大/小的东西，并更新 XXX。」（有时「XXX 中最大/小的东西」可以优化，比如用优先队列维护）

二者的区别在于一种是离线的，先处理后选择；一种是在线的，边处理边选择。

### 排序解法

用排序法常见的情况是输入一个包含几个（一般一到两个）权值的数组，通过排序然后遍历模拟计算的方法求出最优值。

### 后悔解法

思路是无论当前的选项是否最优都接受，然后进行比较，如果选择之后不是最优了，则反悔，舍弃掉这个选项；否则，正式接受。如此往复。

## 例题

### 邻项交换法的例题

> [!NOTE] **[NOIP 2012 国王游戏](https://vijos.org/p/1779)**
> 
> 恰逢 H 国国庆，国王邀请 n 位大臣来玩一个有奖游戏。首先，他让每个大臣在左、右手上面分别写下一个整数，国王自己也在左、右手上各写一个整数。然后，让这 n 位大臣排成一排，国王站在队伍的最前面。排好队后，所有的大臣都会获得国王奖赏的若干金币，每位大臣获得的金币数分别是：排在该大臣前面的所有人的左手上的数的乘积除以他自己右手上的数，然后向下取整得到的结果。
>
> 国王不希望某一个大臣获得特别多的奖赏，所以他想请你帮他重新安排一下队伍的顺序，使得获得奖赏最多的大臣，所获奖赏尽可能的少。注意，国王的位置始终在队伍的最前面。

> [!TIP] **解题思路**
> 
> 设排序后第 $i$ 个大臣左右手上的数分别为 $a_i, b_i$。考虑通过邻项交换法推导策略。
> 
> 用 $s$ 表示第 $i$ 个大臣前面所有人的 $a_i$ 的乘积，那么第 $i$ 个大臣得到的奖赏就是 $\dfrac{s} {b_i}$，第 $i + 1$ 个大臣得到的奖赏就是 $\dfrac{s \cdot a_i} {b_{i+1}}$。
>
> 如果我们交换第 $i$ 个大臣与第 $i + 1$ 个大臣，那么此时的第 $i$ 个大臣得到的奖赏就是 $\dfrac{s} {b_{i+1}}$，第 $i + 1$ 个大臣得到的奖励就是 $\dfrac{s \cdot a_{i+1}} {b_i}$。
>
> 如果交换前更优当且仅当
>
>   $$
>   \max \left(\dfrac{s} {b_i}, \dfrac{s \cdot a_i} {b_{i+1}}\right)  < \max \left(\dfrac{s} {b_{i+1}}, \dfrac{s \cdot a_{i+1}} {b_i}\right)
>   $$
> 
>   提取出相同的 $s$ 并约分得到
>
>   $$
>   \max \left(\dfrac{1} {b_i}, \dfrac{a_i} {b_{i+1}}\right)  < \max \left(\dfrac{1} {b_{i+1}}, \dfrac{a_{i+1}} {b_i}\right)
>   $$
> 
>   然后分式化成整式得到
> 
>   $$
>   \max (b_{i+1}, a_i\cdot b_i)  < \max (b_i, a_{i+1}\cdot b_{i+1})
>   $$
> 
>   实现的时候我们将输入的两个数用一个结构体来保存并重载运算符：


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


### 后悔法的例题

> [!NOTE] **[「USACO09OPEN」工作调度 Work Scheduling](https://www.luogu.com.cn/problem/P2949)**
> 
> 约翰的工作日从 $0$ 时刻开始，有 $10^9$ 个单位时间。在任一单位时间，他都可以选择编号 $1$ 到 $N$ 的 $N(1 \leq N \leq 10^5)$ 项工作中的任意一项工作来完成。工作 $i$ 的截止时间是 $D_i(1 \leq D_i \leq 10^9)$，完成后获利是 $P_i( 1\leq P_i\leq 10^9 )$。在给定的工作利润和截止时间下，求约翰能够获得的利润最大为多少。

> [!TIP] **解题思路**
> 
> 1. 先假设每一项工作都做，将各项工作按截止时间排序后入队；
> 
> 2.  在判断第 i 项工作做与不做时，若其截至时间符合条件，则将其与队中报酬最小的元素比较，若第 i 项工作报酬较高（后悔），则 `ans += a[i].p - q.top()`。  
> 
> 用优先队列（小根堆）来维护队首元素最小。

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

## 习题

- [P1209\[USACO1.3\]修理牛棚 Barn Repair - 洛谷](https://www.luogu.com.cn/problem/P1209)
- [P2123 皇后游戏 - 洛谷](https://www.luogu.com.cn/problem/P2123)
- [LeetCode 上标签为算法的题目](https://leetcode-cn.com/tag/greedy/)

### 基本贪心

> [!NOTE] **[AcWing 1348. 搭配牛奶](https://www.acwing.com/problem/content/1350/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 简单学习下 struct 比较级函数

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 5010;

int n, m;
struct Milk {
    int p, a;
    bool operator< (const Milk& t) const {
        return p < t.p;
    }
}milk[N];

int main() {
    cin >> n >> m;
    for (int i = 0; i < m; ++ i ) cin >> milk[i].p >> milk[i].a;
    sort(milk, milk + m);
    
    int res = 0;
    for (int i = 0; i < m && n; ++ i ) {
        int add = min(n, milk[i].a);
        n -= add;
        res += add * milk[i].p;
    }
    cout << res << endl;
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

> [!NOTE] **[AcWing 1349. 修理牛棚](https://www.acwing.com/problem/content/1351/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 补集的思想 先把0～c-1所有的用一个木板覆盖 随后断开

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 210;

int m, s, c;
int a[N], b[N]; // V 表示间隙

int main() {
    cin >> m >> s >> c;
    for (int i = 0; i < c; ++ i ) cin >> a[i];
    sort(a, a + c);
    
    int res = a[c - 1] - a[0] + 1;
    
    for (int i = 1; i < c; ++ i ) b[i] = a[i] - a[i - 1] - 1;
    sort(b + 1, b + c, greater<int>());
    
    // 至多断开m - 1次
    for (int i = 1; i <= m - 1 && i < c; ++ i )
        res -= b[i];
    
    cout << res << endl;
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

> [!NOTE] **[Luogu [USACO07MAR]Face The Right Way G](https://www.luogu.com.cn/problem/P2882)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 反转问题/开关问题
> 
> 贪心 动态维护

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 5010;

int n;
int d[N], f[N];

int get(int k) {
    memset(f, 0, sizeof f);
    
    // s 维护会影响到i的点击数
    int s = 0, res = 0;
    for (int i = 0; i + k <= n; ++ i ) {
        // 第i头牛 + 之前能影响到它的点击数
        // &1 说明当前朝后 需要点击
        if ((d[i] + s) & 1)
            f[i] = 1, res ++ ;
        
        s += f[i];
        if (i - k + 1 >= 0)
            s -= f[i - k + 1];
    }
    
    // 检查最后一段
    for (int i = n - k + 1; i < n; ++ i ) {
        if ((d[i] + s) & 1)
            return -1;
        if (i - k + 1 >= 0)
            s -= f[i - k + 1];
    }
    return res;
}

int main() {
    cin >> n;
    
    for (int i = 0; i < n; ++ i ) {
        char c;
        cin >> c;
        if (c == 'F')
            d[i] = 0;
        else
            d[i] = 1;
    }
    
    int K = 1, M = n;
    for (int k = 1; k <= n; ++ k ) {
        int m = get(k);
        if (m >= 0 && m < M)
            M = m, K = k;
    }
    cout << K << ' ' << M << endl;
    
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

> [!NOTE] **[LeetCode 1363. 形成三的最大倍数](https://leetcode-cn.com/problems/largest-multiple-of-three/)**
> 
> 题意: 
> 
> 给你一个整数数组 `digits`，你可以通过按任意顺序连接其中某些数字来形成 3 的倍数，请你返回所能得到的最大的 3 的倍数。

> [!TIP] **思路**
> 
> 对所有整数求和 随后依据mod3得到的结果去除一些整数 再生成res

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    string largestMultipleOfThree(vector<int>& digits) {
        vector<int> cnt(10);
        int sum = 0;
        for (auto v : digits) {
            ++cnt[v];
            sum += v;
        }
        bool f = true;
        if (sum % 3 == 0) {
        } else if (sum % 3 == 1) {
            if (cnt[1])
                --cnt[1];
            else if (cnt[4])
                --cnt[4];
            else if (cnt[7])
                --cnt[7];
            else if (cnt[2] >= 2)
                cnt[2] -= 2;
            else if (cnt[5] >= 2)
                cnt[5] -= 2;
            else if (cnt[8] >= 2)
                cnt[8] -= 2;
            else
                f = false;
        } else if (sum % 3 == 2) {
            if (cnt[2])
                --cnt[2];
            else if (cnt[5])
                --cnt[5];
            else if (cnt[8])
                --cnt[8];
            else if (cnt[1] >= 2)
                cnt[1] -= 2;
            else if (cnt[4] >= 2)
                cnt[4] -= 2;
            else if (cnt[7] >= 2)
                cnt[7] -= 2;
            else
                f = false;
        }

        string res;
        if (!f) {
            if (cnt[0]) res.push_back('0');
            return res;
        }
        for (int v = 9; v > 0; --v) {
            while (cnt[v]--) { res.push_back('0' + v); }
        }
        if (res.size())
            while (cnt[0]--) res.push_back('0');
        else if (cnt[0])
            res.push_back('0');
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

> [!NOTE] **[Codeforces Hometask](http://codeforces.com/problemset/problem/214/B)**
> 
> 题意: 
> 
> 比上题条件更严格些

> [!TIP] **思路**
> 
> 类似 **[LeetCode 1363. 形成三的最大倍数](https://leetcode-cn.com/problems/largest-multiple-of-three/)**
> 
> 分情况讨论

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: B. Hometask
// Contest: Codeforces - Codeforces Round #131 (Div. 2)
// URL: https://codeforces.com/problemset/problem/214/B
// Memory Limit: 256 MB
// Time Limit: 2000 ms

#include <bits/stdc++.h>
using namespace std;

const static int N = 10;

int n, sum = 0;
int cnt[N], has[N];

int main() {
    cin >> n;
    for (int i = 0, x; i < n; ++i)
        cin >> x, cnt[x]++, sum += x;

    if (cnt[0] == 0) {
        cout << -1 << endl;
        return 0;
    }

    if (sum % 3 == 1) {
        if (cnt[1])
            sum -= 1, cnt[1]--;
        else if (cnt[4])
            sum -= 4, cnt[4]--;
        else if (cnt[7])
            sum -= 7, cnt[7]--;
        else if (cnt[2] >= 2)
            sum -= 4, cnt[2] -= 2;
        else if (cnt[2] && cnt[5])
            sum -= 7, cnt[2]--, cnt[5]--;
        else if (cnt[5] >= 2)
            sum -= 10, cnt[5] -= 2;
        else if (cnt[5] && cnt[8])
            sum -= 13, cnt[5]--, cnt[8]--;
        else if (cnt[8] >= 2)
            sum -= 16, cnt[8] -= 2;
        else {
            cout << 0 << endl;  // ATTENTION
            return 0;
        }
    }
    if (sum % 3 == 2) {
        if (cnt[2])
            sum -= 2, cnt[2]--;
        else if (cnt[5])
            sum -= 5, cnt[5]--;
        else if (cnt[8])
            sum -= 8, cnt[8]--;
        else if (cnt[1] >= 2)
            sum -= 2, cnt[1] -= 2;
        else if (cnt[1] && cnt[4])
            sum -= 5, cnt[1]--, cnt[4]--;
        else if (cnt[3] && cnt[5])
            sum -= 8, cnt[3]--, cnt[5]--;
        else if (cnt[4] >= 2)
            sum -= 8, cnt[4] -= 2;
        else if (cnt[7] >= 2)
            sum -= 14, cnt[7] -= 2;
        else {
            cout << 0 << endl;
            return 0;
        }
    }

    string res;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < cnt[i]; ++j)
            res.push_back('0' + i);

    while (res.size() > 1 && res.back() == '0')
        res.pop_back();
    reverse(res.begin(), res.end());
    cout << res << endl;

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


> [!NOTE] **[LeetCode 1402. 做菜顺序](https://leetcode-cn.com/problems/reducing-dishes/)**
> 
> 题意: 
> 
> 做出每道菜的时间都是 1 单位时间。一道菜的 「喜爱时间」系数定义为烹饪这道菜以及之前每道菜所花费的时间乘以这道菜的满意程度，也就是 $time[i]*satisfaction[i]$ 。顺序随便排，可以舍弃任意菜。
> 
> 返回做完所有菜 「喜爱时间」总和的最大值为多少。

> [!TIP] **思路**
> 
> 贪心 每次选满意度最高的 然后向后移动(sum) 相当于后面的菜全部又加了后面的菜满意度之和: v + sum
> 
> 坑点在于题意的理解 以及基于贪心思考的实现转化

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int maxSatisfaction(vector<int>& satisfaction) {
        sort(satisfaction.begin(), satisfaction.end());
        int n = satisfaction.size(), res = 0;
        for (int i = n - 1, tot = 0, sum = 0; i >= 0; -- i ) {
            tot = tot + sum + satisfaction[i];
            if (res < tot)
                res = tot;
            sum += satisfaction[i];
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

> [!NOTE] **[LeetCode 1414. 和为 K 的最少斐波那契数字数目](https://leetcode-cn.com/problems/find-the-minimum-number-of-fibonacci-numbers-whose-sum-is-k/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 返回和为 k 的斐波那契数字的最少数目，其中，每个斐波那契数字都可以被使用多次。

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int findMinFibonacciNumbers(int k) {
        vector<int> fib;
        fib.push_back(1);
        fib.push_back(2);
        int a = 1, b = 2, c = 3;
        while (c <= k) {
            fib.push_back(c);
            a = b;
            b = c;
            c = a + b;
        }
        int n = fib.size(), res = 0;
        // === Solutiong 1
        for (int i = n - 1; i >= 0; --i) {
            if (k > fib[i]) {
                ++res;
                k -= fib[i];
            } else if (k == fib[i]) {
                ++res;
                break;
            }
        }
        // ===
        // === Solution 2
        while (k > 0) {  // 从大到小贪心选取
            if (k >= fi[i]) k -= fi[i], ++res;
            --i;
        }
        // ===
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

> [!NOTE] **[LeetCode 1536. 排布二进制网格的最少交换次数](https://leetcode-cn.com/problems/minimum-swaps-to-arrange-a-binary-grid/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 要求：主对角线以上的格子全0
> 
> 考虑统计每一行右侧连续0个数 记录分界位置于 rec
> 
> 随后贪心：从后面找到最近可满足的 【因为从前往后的要求逐级降低】 所以次近的

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int minSwaps(vector<vector<int>>& grid) {
        int n = grid.size();
        vector<int> rec(n);
        for (int i = 0; i < n; ++i) {
            int cnt = 0;
            for (int j = n - 1; j > 0; --j) {
                if (grid[i][j])
                    break;
                else
                    ++cnt;
            }
            rec[i] = n - 1 - cnt;
        }
        int res = 0;
        for (int i = 0; i < n; ++i) {
            if (rec[i] <= i) continue;
            int j = i + 1;
            while (j < n && rec[j] > i) ++j;
            if (j >= n) return -1;
            res += j - i;
            int t = rec[j];
            for (int k = j; k > i; k--) { rec[k] = rec[k - 1]; }
            rec[i] = t;
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

> [!NOTE] **[LeetCode 1691. 堆叠长方体的最大高度](https://leetcode-cn.com/problems/maximum-height-by-stacking-cuboids/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 对每一个方块贪心排列
> 
> 【有去除长度限制 可任意翻转的版本 TODO 解决这个题】

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int maxHeight(vector<vector<int>>& cuboids) {
        for (auto & c : cuboids) sort(c.begin(), c.end());
        sort(cuboids.begin(), cuboids.end(), greater<vector<int>>());
        int n = cuboids.size();
        int res = 0;
        vector<int> f(n);
        for (int i = 0; i < n; ++ i ) {
            f[i] = cuboids[i][2];
            for (int j = 0; j < i; ++ j )
                if (cuboids[j][0] >= cuboids[i][0] && cuboids[j][1] >= cuboids[i][1] && cuboids[j][2] >= cuboids[i][2])
                    f[i] = max(f[i], f[j] + cuboids[i][2]);
            res = max(res, f[i]);
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

> [!NOTE] **[LeetCode 1717. 删除子字符串的最大得分](https://leetcode-cn.com/problems/maximum-score-from-removing-substrings/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 贪心 记忆处理技巧

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int maximumGain(string s, int x, int y) {
        // 确保最终优先选择的都是'ab'
        if (x < y) {
            swap(x, y);
            for (auto & c : s) {
                if (c == 'a') c = 'b';
                else if (c == 'b') c = 'a';
            }
        }

        int res = 0;
        for (int i = 0; i < s.size(); ++ i ) {
            if (s[i] != 'a' && s[i] != 'b') continue;
            int j = i + 1;
            while (j < s.size() && (s[j] == 'a' || s[j] == 'b')) ++ j ;
            // 全是 a 或 b 的一段
            int a = 0, b = 0, c = 0;
            for (int k = j - 1, t = 0; k >= i; -- k )
                if (s[k] == 'a') {
                    ++ a ;
                    // 后方有未使用的b
                    if (t) {
                        ++ c ;
                        -- t ;
                    }
                } else {
                    ++ b ;
                    ++ t ;
                }
            
            res += c * x + (min(a, b) - c) * y;
            i = j - 1;
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

> [!NOTE] **[LeetCode 1798. 你能构造出连续值的最大数目](https://leetcode-cn.com/problems/maximum-number-of-consecutive-values-you-can-make/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 类似的总结 ==> 跳跃游戏 TODO

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int getMaximumConsecutive(vector<int>& coins) {
        sort(coins.begin(), coins.end());
        int lst = 0, n = coins.size();
        for (int i = 0; i < n; ++ i )
            if (coins[i] <= lst + 1)
                lst += coins[i];
            else
                break;
        return lst + 1;
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

> [!NOTE] **[LeetCode 2086. 从房屋收集雨水需要的最少水桶数](https://leetcode-cn.com/problems/minimum-number-of-buckets-required-to-collect-rainwater-from-houses/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> **尽量选择一个房屋右侧的点来放水桶**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

标准解是贪心在右侧选 且是判断 `H` 的位置

```cpp
class Solution {
public:
    int minimumBuckets(string street) {
        int n = street.size(), res = 0;
        vector<bool> st(n);
        for (int i = 0; i < n; ++ i )
            // trick: 思维 先考虑 `H`
            if (street[i] == 'H') {
                if (i - 1 >= 0 && st[i - 1])
                    continue;
                // 优先放右边
                if (i + 1 < n && street[i + 1] == '.')
                    st[i + 1] = true, res ++ ;
                else if (i - 1 >= 0 && street[i - 1] == '.')
                    st[i - 1] = true, res ++ ;
                else
                    return -1;
            }
        
        return res;
    }
};
```

##### **C++ 个人写法**

```cpp
class Solution {
public:
    int minimumBuckets(string street) {
        int n = street.size(), res = 0;
        vector<bool> st(n);
        for (int i = 1; i < n - 1; ++ i )
            if (street[i] == '.' && street[i - 1] == 'H' && street[i + 1] == 'H')
                street[i] = '*', st[i - 1] = st[i + 1] = true, res ++ ;
        for (int i = 2; i < n - 2; ++ i )
            if (street[i] == '*' && street[i - 2] == '*' && street[i + 2] == '*')
                street[i] = '.', res -- ;
        for (int i = 0; i < n; ++ i )
            if (street[i] == '.') {
                bool need = false;
                if (i - 1 >= 0 && street[i - 1] == 'H' && !st[i - 1])
                    need = true, st[i - 1] = true;
                if (i + 1 < n && street[i + 1] == 'H' && !st[i + 1])
                    need = true, st[i + 1] = true;
                if (need)
                    res ++ ;
            }
        for (int i = 0; i < n; ++ i )
            if (street[i] == 'H' && !st[i])
                return -1;
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

> [!NOTE] **[Codeforces C. The Child and Toy](https://codeforces.com/problemset/problem/437/C)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 贪心 证明 思路总结

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: C. The Child and Toy
// Contest: Codeforces - Codeforces Round #250 (Div. 2)
// URL: https://codeforces.com/problemset/problem/437/C
// Memory Limit: 256 MB
// Time Limit: 1000 ms

#include <bits/stdc++.h>
using namespace std;

using PII = pair<int, int>;
const int N = 1010, M = 4010;

int n, m;
int h[N], e[M], ne[M], idx;
int a[N];
PII w[N];
bool st[N];

void add(int a, int b) { e[idx] = b, ne[idx] = h[a], h[a] = idx++; }

int main() {
    cin >> n >> m;

    for (int i = 1; i <= n; ++i)
        cin >> a[i];

    for (int i = 1; i <= n; ++i)
        w[i] = {a[i], i};

    sort(w + 1, w + n + 1);

    memset(h, -1, sizeof h);
    while (m--) {
        int a, b;
        cin >> a >> b;
        add(a, b), add(b, a);
    }

    int res = 0;
    // 直觉 贪心先找权值最大的点 删掉它 消耗是其周围的边
    for (int i = n; i >= 1; --i) {
        auto [v, id] = w[i];
        for (int j = h[id]; ~j; j = ne[j]) {
            int k = e[j], c = a[k];
            if (!st[k])
                res += c;
        }
        st[id] = true;
    }

    cout << res << endl;

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

> [!NOTE] **[Codeforces C. Table Decorations](http://codeforces.com/problemset/problem/478/C)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 数学 思维题 思路

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: C. Table Decorations
// Contest: Codeforces - Codeforces Round #273 (Div. 2)
// URL: http://codeforces.com/problemset/problem/478/C
// Memory Limit: 256 MB
// Time Limit: 1000 ms

#include <bits/stdc++.h>
using namespace std;

// 贪心题 思维
// 考虑 r,g,b 装饰桌子 每个桌子三个气球且颜色不能完全相同
// 求最多可以装饰多少个桌子

using LL = long long;

int main() {
    LL r, g, b;
    cin >> r >> g >> b;
    // cout << min(min(min((r + g + b) / 3, r + g), r + b), b + g) << endl;

    vector<LL> ve{r, g, b};
    sort(ve.begin(), ve.end());

    if (ve[0] + ve[1] <= ve[2] / 2)
        cout << ve[0] + ve[1] << endl;
    else
        cout << (ve[0] + ve[1] + ve[2]) / 3 << endl;

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

> [!NOTE] **[Codeforces GukiZ hates Boxes](https://codeforces.com/problemset/problem/551/C)**
> 
> 题意: 
> 
> 有 $n$ 个位置，第 $i$ 个位置上有 $a_i$ 个箱子。
> 
> 有 $m$ 个人，开始在 $0$ 位置，每一秒钟每个人都可以选择搬走自己位置上的一个箱子或向前走一步。
> 
> 问最少需要多少时间才可以将箱子全部搬完。
> 
> $n,m\le 10^5,a_i\le 10^9$

> [!TIP] **思路**
> 
> 优先搬末尾，重点在于贪心推导

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: C. GukiZ hates Boxes
// Contest: Codeforces - Codeforces Round #307 (Div. 2)
// URL: https://codeforces.com/problemset/problem/551/C
// Memory Limit: 256 MB
// Time Limit: 2000 ms

#include <bits/stdc++.h>
using namespace std;

// 一开始题意理解有误，其实题意中的“移除盒子”就是在原地就直接数量减一即可。
// 而且每个人可以执行移除操作无限次（只要在时间范围内），可无限左右移动
// ==>
// 在这个前提下，我们显然可以得到

using LL = long long;
const static int N = 1e5 + 10;

int n, m;
int a[N];

// 检查 mid 时间内能否移除所有盒子
bool check(LL mid) {
    // 在此前提下，显然可以假定每个人都有 mid 的时间
    // 【每个人拿尽可能靠右的箱子】==》重要的贪心推导

    vector<int> xs;  // 标记所有有箱子的位置
    static LL t[N];
    for (int i = 1; i <= n; ++i) {
        t[i] = a[i];
        if (t[i])
            xs.push_back(i);
    }

    for (int i = 1; i <= m && !xs.empty(); ++i) {
        if (mid < xs.back())
            return false;
        // 一直到到达末尾，可以用于搬运盒子的次数【重要贪心：找末尾】
        LL left = mid - xs.back();
        while (xs.size() && left) {
            LL move = min(left, t[xs.back()]);
            left -= move;
            if (move == t[xs.back()])
                xs.pop_back();
            else
                t[xs.back()] -= move;
        }
    }
    return xs.empty();
}

int main() {
    cin >> n >> m;
    for (int i = 1; i <= n; ++i)
        cin >> a[i];

    // 显然最小时间有二分性质
    LL l = 0, r = 1e18;
    while (l < r) {
        LL mid = l + r >> 1;
        if (!check(mid))
            l = mid + 1;
        else
            r = mid;
    }
    cout << l << endl;

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

### 排序不等式

> [!NOTE] **[AcWing 1395. 产品处理](https://www.acwing.com/problem/content/1397/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> **思维题**
> 
> Q1 
> 
> Q2 需要排序不等式

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 1010;

int n, m1, m2;
int a[N], b[N];
struct Node {
    int finish_time, cost;
    bool operator< (const Node & t) const {
        return finish_time > t.finish_time;
    }
};

void work(int m, int f[]) {
    priority_queue<Node> heap;
    for (int i = 0; i < m; ++ i ) {
        int cost;
        cin >> cost;
        // 加入的是 【洗完下一件衣服最早结束的时间,单个时间消耗】
        heap.push({cost, cost});
    }
    
    for (int i = 1; i <= n; ++ i ) {
        auto t = heap.top(); heap.pop();
        f[i] = t.finish_time;
        t.finish_time += t.cost;
        heap.push(t);
    }
}

int main() {
    cin >> n >> m1 >> m2;
    work(m1, a);
    work(m2, b);
    
    // attention 互补这样
    // ------  ---
    // ----  -----
    // -- --------
    // - ---------
    int res = 0;
    for (int i = 1, j = n; i <= n; ++ i , -- j )
        res = max(res, a[i] + b[j]);
    cout << a[n] << ' ' << res << endl;
    
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

> [!NOTE] **[Codeforces Mafia](http://codeforces.com/problemset/problem/348/A)**
> 
> 题意: 
> 
> 第 $i$ 人至少参与游戏 $a[i]$ 次，每次游戏必有一个不参与的裁判
> 
> 求最少举行多少场游戏

> [!TIP] **思路**
> 
> 某人当裁判相当于只有它减少，易知类似互补的思想

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: A. Mafia
// Contest: Codeforces - Codeforces Round #202 (Div. 1)
// URL: https://codeforces.com/problemset/problem/348/A
// Memory Limit: 256 MB
// Time Limit: 2000 ms

#include <bits/stdc++.h>
using namespace std;

using LL = long long;
const static int N = 1e5 + 10;

int n;
LL a[N];

int main() {
    cin >> n;
    LL s = 0, mx = 0;
    for (int i = 1; i <= n; ++i)
        cin >> a[i], s += a[i], mx = max(mx, a[i]);

    LL l = mx, r = s;
    while (l < r) {
        LL m = l + r >> 1;
        if (m * n - s < m)
            l = m + 1;
        else
            r = m;
    }
    cout << l << endl;

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


> [!NOTE] **[AcWing 913. 排队打水](https://www.acwing.com/problem/content/description/915/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 排序不等式
> 
> 显然耗时最短的越先打水

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

int n, x;

int main() {
    cin >> n;
    vector<int> ve;
    for (int i = 0; i < n; ++i) {
        cin >> x;
        ve.push_back(x);
    }
    sort(ve.begin(), ve.end());
    long long res = 0;
    for (auto x : ve) { res += x * (--n); }
    cout << res << endl;
}
```

##### **Python**

```python
# 直觉就是：让权重小的人排在前面；最优解是：从小到大排序，那么总时间最小。
# 证明：(用调整法/反证法证明)
# 假设最优解不是从小到大排序，那会存在相邻两个点是逆序的；交换两个位置，交换前的值 > 交换后的值。说明交换后总时间会降低。那就和假设矛盾，所以最优解一定是从小到大的排序。

if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().split()))

    arr.sort()
    res = 0
    for i in range(n):
        res += arr[i] * (n - i - 1)
    print(res)
```

<!-- tabs:end -->
</details>

<br>

* * *


### 绝对值不等式

> [!NOTE] **[AcWing 104. 货仓选址](https://www.acwing.com/problem/content/106/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 绝对值不等式
> 
> 经典 选择中间的两个点之间的任意位置

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

int n, x;

int main() {
    cin >> n;
    vector<int> ve;
    for (int i = 0; i < n; ++i) {
        cin >> x;
        ve.push_back(x);
    }
    sort(ve.begin(), ve.end());
    // n为奇数 中间为 3/2=1 (0,1,2)  n为偶数 中间为 4/2=2 (0,1,2,3)
    int res = 0;
    for (auto v : ve) res += abs(v - ve[n / 2]);
    cout << res << endl;
}
```

##### **Python**

```python
if __name__ == '__main__':
    n = int(input())
    nums = list(map(int, input().split()))
    # 踩坑：不是nums=nums.sort()
    nums.sort()
    res = 0
    m = (n - 1) // 2
    for i in range(n):
        res += abs(nums[i] - nums[m])
    print(res)

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[AcWing 105. 七夕祭](https://www.acwing.com/problem/content/107/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 货仓选址进阶
> 
> 操作行和列互不影响
> 
> 则总体最小 为 操作行最+操作列最小
> 
> 模型：环形纸牌 邮递员问题 视频关于n-1个等式的转化和分解 ==> 取中位数

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include<bits/stdc++.h>
using namespace std;

using LL = long long;

const int N = 100010;

int row[N], col[N], s[N], c[N];

LL work(int n, int a[]) {
    for (int i = 1; i <= n; ++ i ) s[i] = s[i - 1] + a[i];
    if (s[n] % n) return -1;
    
    int avg = s[n] / n;
    // 求Ci
    c[1] = 0;
    for (int i = 2; i <= n; ++ i ) c[i] = s[i - 1] - (i - 1) * avg;
    
    sort(c + 1, c + n + 1);
    LL res = 0;
    for (int i = 1; i <= n; ++ i ) res += abs(c[i] - c[(n + 1) / 2]);
    return res;
}

int main() {
    int n, m, cnt;
    cin >> n>> m >> cnt;
    while (cnt -- ) {
        int x, y;
        cin >> x >> y;
        ++ row[x], ++ col[y];
    }
    LL r = work(n, row);
    LL c = work(m, col);
    
    if (r != -1 && c != -1) cout << "both "<< r + c << endl;
    else if (r != -1) cout << "row " << r << endl;
    else if (c != -1) cout << "column " << c << endl;
    else cout << "impossible" << endl;

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

> [!NOTE] **[LeetCode 462. 最少移动次数使数组元素相等 II](https://leetcode-cn.com/problems/minimum-moves-to-equal-array-elements-ii/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ yxc**

```cpp
// yxc
class Solution {
public:
    int minMoves2(vector<int>& nums) {
        int n = nums.size();
        sort(nums.begin(), nums.end());
        int res = 0;
        for (int i = 0; i < n; i ++ )
            res += abs(nums[i] - nums[n / 2]);
        return res;
    }
};
```

##### **C++**

```cpp
class Solution {
public:
    int minMoves2(vector<int>& nums) {
        int res = 0, l = 0, r = nums.size() - 1;
        sort(nums.begin(), nums.end());
        while (l < r) {
            res += nums[r -- ] - nums[l ++ ];
        }
        return res;
    }
};
```

##### **C++ nth 略**

```cpp

```


##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 1703. 得到连续 K 个 1 的最少相邻交换次数](https://leetcode-cn.com/problems/minimum-adjacent-swaps-for-k-consecutive-ones/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 所有1坐标 如何移动使得其新的坐标成为一个等差数列（公差1）
> 
> 考虑计算答案时是绝对值求和取 min ，转化映射后即绝对值不等式
> 
> 做一个映射：`ai' = aix - i`

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 交换过程中1的相对位置必然不变
    // 绝对值不等式
    using LL = long long;
    int minMoves(vector<int>& nums, int k) {
        vector<int> ve;
        for (int i = 0; i < nums.size(); ++ i )
            if (nums[i])
                ve.push_back(i - ve.size());
        int n = ve.size();
        vector<LL> s(n + 1);
        for (int i = 1; i <= n; ++ i ) s[i] = s[i - 1] + ve[i - 1];
        LL res = 1e18;
        for (int i = k; i <= n; ++ i ) {
            int l = i - k + 1, r = i;
            int mid = (l + r) / 2;
            LL x = ve[mid - 1];
            LL left = x * (mid - l) - (s[mid - 1] - s[l - 1]);
            LL right = (s[r] - s[mid]) - x * (r - mid);
            res = min(res, left + right);
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

> [!NOTE] **[LeetCode 2033. 获取单值网格的最小操作数](https://leetcode-cn.com/problems/minimum-operations-to-make-a-uni-value-grid/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> - 自己有较直观的解法
> 
> - 实际上 这就是一个打水问题 要求【所有点到某点的距离和最小 该点即为中点】
> 
>   需要增强对该结论的敏感程度

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 直观解法**

```cpp
class Solution {
public:
    using LL = long long;
    
    int n, m;
    
    int minOperations(vector<vector<int>>& grid, int x) {
        this->n = grid.size(), this->m = grid[0].size();
        vector<int> nums;
        for (int i = 0; i < n; ++ i )
            for (int j = 0; j < m; ++ j )
                nums.push_back(grid[i][j]);
        sort(nums.begin(), nums.end());
        
        int len = nums.size(), v = nums[0];
        {
            nums[0] = 0;
            for (int i = 1; i < len; ++ i ) {
                nums[i] -= v;
                if (nums[i] % x)
                    return -1;
                nums[i] /= x;
            }
        }
        
        vector<int> l(len), r(len);
        for (int i = 1; i < len; ++ i )
            l[i] = l[i - 1] + nums[i - 1];
        for (int i = len - 2; i >= 0; -- i )
            r[i] = r[i + 1] + nums[i + 1];
        
        int res = 2e9;
        for (int i = 0; i < len; ++ i ) {
            int t = 0;
            if (i - 1 >= 0)
                t += i * nums[i] - l[i];
            if (i + 1 < len)
                t += r[i] - (len - i - 1) * nums[i];
            res = min(res, t);
        }
        return res;
    }
};
```

##### **C++ 绝对值不等式**

```cpp
class Solution {
public:
    using LL = long long;
    
    int n, m;
    
    int minOperations(vector<vector<int>>& grid, int x) {
        this->n = grid.size(), this->m = grid[0].size();
        vector<int> nums;
        for (int i = 0; i < n; ++ i )
            for (int j = 0; j < m; ++ j )
                nums.push_back(grid[i][j]);
        
        // 使用 nth_element 无需排序
        // sort(nums.begin(), nums.end());
        
        int len = nums.size(), v = nums[0];
        {
            nums[0] = 0;
            for (int i = 1; i < len; ++ i ) {
                nums[i] -= v;
                if (nums[i] % x)
                    return -1;
                nums[i] /= x;
            }
        }

        // ATTENTION 显然此类问题可以直接找中点
        //   加强思维上的敏感度
        int mid = n * m / 2;
        nth_element(nums.begin(), nums.begin() + mid, nums.end());

        int res = 0;
        for (auto v : nums)
            res += abs(v - nums[mid]);
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


> [!NOTE] **[Codeforces B. Polo the Penguin and Matrix](https://codeforces.com/problemset/problem/289/B)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 同 [LeetCode 2033. 获取单值网格的最小操作数](https://leetcode-cn.com/problems/minimum-operations-to-make-a-uni-value-grid/)
> 
> 很多博客并未提及为何可以使用中位数，直观来看使用中位数无法确保其他数可以有效转变为该中位数。
> 
> 但分析推导易知，矩阵中所有的数字对 d 取模的值都应相等，否则不合法。
> 
> 在合法的情况下所有数字必然可以通过 + / - d 的办法得到统一。具体参见 https://codeforces.com/blog/entry/7241

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: B. Polo the Penguin and Matrix
// Contest: Codeforces - Codeforces Round #177 (Div. 2)
// URL: https://codeforces.com/problemset/problem/289/B
// Memory Limit: 256 MB
// Time Limit: 2000 ms

#include <bits/stdc++.h>
using namespace std;

const int N = 10010;

int n, m, d;
int a[N];

int main() {
    cin >> n >> m >> d;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            cin >> a[i * m + j];
    sort(a, a + n * m);

    int t = n * m / 2, res = 0;
    bool f = true;
    for (int i = 0; i < n * m; ++i) {
        if (abs(a[t] - a[i]) % d) {
            f = false;
            break;
        }
        res += abs(a[t] - a[i]) / d;
    }
    cout << (f ? res : -1) << endl;
}
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

### 区间问题

> [!NOTE] **[AcWing 803. 区间合并](https://www.acwing.com/problem/content/805/)**
> 
> 题意: [LeetCode 56. 合并区间](https://leetcode-cn.com/problems/merge-intervals/)

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;

typedef pair<int, int> PII;

void merge(vector<PII> &segs) {
    vector<PII> res;

    sort(segs.begin(), segs.end());

    int st = -2e9, ed = -2e9;
    for (auto seg : segs)
        if (ed < seg.first) {
            if (st != -2e9) res.push_back({st, ed});
            st = seg.first, ed = seg.second;
        } else
            ed = max(ed, seg.second);

    if (st != -2e9) res.push_back({st, ed});

    segs = res;
}

int main() {
    int n;
    scanf("%d", &n);

    vector<PII> segs;
    for (int i = 0; i < n; i++) {
        int l, r;
        scanf("%d%d", &l, &r);
        segs.push_back({l, r});
    }

    merge(segs);

    cout << segs.size() << endl;

    return 0;
}
```

##### **Python**

```python
# 按照区间左端点大小排序
if __name__ == '__main__':
    n = int(input())
    intervals = [list(map(int, input().split())) for _ in range(n)]

    res = []
    intervals.sort()
    l, r = intervals[0][0], intervals[0][1]
    for i in range(1, n):
        if intervals[i][0] <= r:
            r = max(intervals[i][1], r)
        else:
            res.append([l, r])
            l, r = intervals[i][0], intervals[i][1]
    res.append([l, r])
    print(len(res))

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 57. 插入区间](https://leetcode-cn.com/problems/insert-interval/)**
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
    vector<vector<int>> insert(vector<vector<int>>& intervals, vector<int>& newInterval) {
        int n = intervals.size();
        vector<vector<int>> res;
        int p = 0;
        while (p < n && intervals[p][1] < newInterval[0]) res.push_back(intervals[p ++ ]);
        while (p < n && intervals[p][0] <= newInterval[1]) {
            newInterval[0] = min(newInterval[0], intervals[p][0]);
            newInterval[1] = max(newInterval[1], intervals[p][1]);
            ++ p ;
        }
        res.push_back(newInterval);
        while (p < n) res.push_back(intervals[p ++ ]);
        return res;
    }
};
```

##### **Python**

```python
class Solution:
    def insert(self, nums1: List[List[int]], nums2: List[int]) -> List[List[int]]:
        nums1.append(nums2)
        nums1.sort()
        left, right = nums1[0][0], nums1[0][1]
        res = []
        for i in range(1, len(nums1)):
            if nums1[i][0] <= right:
                right = max(right, nums1[i][1])
            else:
                res.append([left,right])
                left, right = nums1[i][0], nums1[i][1]
        res.append([left, right])
        return res
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[AcWing 905. 区间选点](https://www.acwing.com/problem/content/907/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 按右端点排序 如果当前 l 比上次标记大则新增

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int inf = 0x3f3f3f3f;
int n, a, b;
int main() {
    cin >> n;
    vector<pair<int, int>> ve;
    for (int i = 0; i < n; ++i) {
        cin >> a >> b;
        ve.push_back({b, a});
    }
    sort(ve.begin(), ve.end());
    int res = 0, mxr = -inf;
    for (auto [r, l] : ve) {
        if (l <= mxr) continue;
        ++res;
        mxr = r;
    }
    cout << res << endl;
}
```

##### **Python**

```python
# 算法过程：
# 1. 将所有的区间按照右端点从小到大排序
# 2. 从前往后依次枚举每个区间，如果当前区间中已经包含，则pass；否则，选择当前区间的右端点
if __name__ == '__main__':
    n = int(input())
    arr = []
    for _ in range(n):
        arr.append(list(map(int, input().split())))
    arr.sort()
    res = 1;
    r = arr[0][1]
    for i in range(n):
        if arr[i][0] > r:
            res += 1
            r = arr[i][1]
    print(res)

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[AcWing 908. 最大不相交区间数量](https://www.acwing.com/problem/content/910/)**
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

const int inf = 0x3f3f3f3f;
int n, a, b;
int main() {
    cin >> n;
    vector<pair<int, int>> ve;
    // r, l
    for (int i = 0; i < n; ++i) {
        cin >> a >> b;
        ve.push_back({b, a});
    }
    sort(ve.begin(), ve.end());
    int res = 0, mxr = -inf;
    for (auto [r, l] : ve) {
        if (l <= mxr) continue;
        ++res;
        mxr = r;
    }
    cout << res << endl;
}
```

##### **Python**

```python
# 算法过程：
# 1. 将所有的区间按照右端点从小到大排序
# 2. 从前往后依次枚举每个区间，

if __name__ == '__main__':
    n = int(input())
    arr = []
    for _ in range(n):
        arr.append(list(map(int, input().split())))
    arr.sort(key=lambda x: x[1])
    res = n;
    r = arr[0][1]
    for i in range(1, n):
        if arr[i][0] <= r:
            res -= 1
        else:
            r = arr[i][1]
    # 等价变换：      
    # for i in range(1, n):
    #    if arr[i][0] > r:
    #        r = arr[i][1]
    #    else:
    #        res -= 1
    print(res)
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[AcWing 906. 区间分组](https://www.acwing.com/problem/content/908/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 按左端点排序 + 小根堆

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int inf = 0x3f3f3f3f;
int n, a, b;

int main() {
    cin >> n;
    vector<pair<int, int>> ve;
    for (int i = 0; i < n; ++i) {
        cin >> a >> b;
        ve.push_back({a, b});
    }
    sort(ve.begin(), ve.end());
    priority_queue<int, vector<int>, greater<int>>
        heap;  //小根堆始终保证所有组中的最小的右端点为根节点
    for (auto [l, r] : ve) {
        if (heap.empty() || heap.top() >= l)
            heap.push(r);
        else {
            heap.pop();
            heap.push(r);
        }
    }
    cout << heap.size() << endl;
}
```

##### **Python**

```python
# 算法过程：
# 1. 将所有区间按照左端点从小到大排序
# 2. 从前向后处理每个区间：判断能否将其放到某个现有的组中（L[i]>Max_R)
#    1) 如果不存在这样的组，则开新组，然后再将其放进去；
#    2）如果存在这样的组，将其放进去，并且更新当前组的Max_R
if __name__ == '__main__':
    n = int(input())
    arr = []
    for i in range(n):
        arr.append(list(map(int, input().split())))
    arr.sort(key=lambda x: x[0])

    import heapq

    h = []
    for start, end in arr:
        if len(h) == 0 or h[0] >= start:
            heapq.heappush(h, end)
        else:
            heapq.heapreplace(h, end)
    print(len(h))
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[AcWing 907. 区间覆盖](https://www.acwing.com/problem/content/909/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 经典贪心 重复做

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int inf = 0x3f3f3f3f;
int n, a, b, s, t;

int main() {
    cin >> s >> t;
    cin >> n;
    vector<pair<int, int>> ve;
    for (int i = 0; i < n; ++i) {
        cin >> a >> b;
        // 略去无关区间
        if (a > t || b < s) continue;
        ve.push_back({a, b});
    }
    sort(ve.begin(), ve.end());
    int m = ve.size();
    // 剩下的区间全部都有经过[s,t]
    int res = 0;
    bool f = false;
    for (int i = 0; i < m; ++i) {
        int j = i, mx = -inf;
        while (j < m && ve[j].first <= s) mx = max(mx, ve[j].second), ++j;
        // 最靠右的位置都不能满足起始位置（因为是按左端点排序
        if (mx < s) break;
        ++res;
        if (mx >= t) {
            f = true;
            break;
        }
        s = mx;
        i = j - 1;
    }
    if (f)
        cout << res << endl;
    else
        cout << -1 << endl;
}
```

##### **Python**

```python
#算法过程：
#1. 将所有区间按照左端点从小到大排序
#2. 从前往后依次枚举每个区间，在所有能覆盖start的区间中，选择右端点最大的区间；然后将start更新为右端点的最大值。
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 352. 将数据流变为多个不相交区间](https://leetcode-cn.com/problems/data-stream-as-disjoint-intervals/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 经典区间问题

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class SummaryRanges {
public:
    /** Initialize your data structure here. */
    typedef long long LL;
    const LL INF = 1e18;
    typedef pair<LL, LL> PLL;
    #define x first
    #define y second
    
    set<PLL> S;

    SummaryRanges() {
        S.insert({-INF, -INF}), S.insert({INF, INF});
    }
    
    void addNum(int x) {
        auto r = S.upper_bound({x, INT_MAX});
        auto l = r;
        -- l;
        if (l->y >= x) return;
        if (l->y == x - 1 && r->x == x + 1)
            S.insert({l->x, r->y}), S.erase(l), S.erase(r);
        else if (l->y == x - 1)
            S.insert({l->x, x}), S.erase(l);
        else if (r->x == x + 1)
            S.insert({x, r->y}), S.erase(r);
        else
            S.insert({x, x});
    }
    
    vector<vector<int>> getIntervals() {
        vector<vector<int>> res;
        for (auto & p : S)
            if (p.x != -INF && p.x != INF)
                res.push_back({(int)p.x, (int)p.y});
        return res;
    }
};

/**
 * Your SummaryRanges object will be instantiated and called as such:
 * SummaryRanges* obj = new SummaryRanges();
 * obj->addNum(val);
 * vector<vector<int>> param_2 = obj->getIntervals();
 */
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 435. 无重叠区间](https://leetcode-cn.com/problems/non-overlapping-intervals/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 按右端点排序

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 归一写法**

```cpp
class Solution {
public:
    int eraseOverlapIntervals(vector<vector<int>>& intervals) {
        int n = intervals.size();
        sort(intervals.begin(), intervals.end(), [](vector<int> a, vector<int> b) {
            return a[1] < b[1];
        });
        int res = 0, mxr = 0xcfcfcfcf;
        for (int i = 0; i < n; ++ i ) {
            if (intervals[i][0] < mxr) continue;
            ++ res;
            mxr = intervals[i][1];
        }
        return n - res;
    }
};
```

##### **C++**

```cpp
class Solution {
public:
    int eraseOverlapIntervals(vector<vector<int>>& intervals) {
        int len = intervals.size();
        if (len <= 1) return 0;
        sort(intervals.begin(), intervals.end(), [](vector<int> a, vector<int> b) {
            return a[1] < b[1];
        });
        int ans = 0, last = intervals[0][1];
        for(int i = 1; i < len; ++ i )
            if (intervals[i][0] < last) ans ++ ;
            else last = intervals[i][1];
        return ans;
    }
};

// yxc
class Solution {
public:
    int eraseOverlapIntervals(vector<vector<int>>& q) {
        sort(q.begin(), q.end(), [](vector<int> a, vector<int> b) {
            return a[1] < b[1];
        });
        if (q.empty()) return 0;
        int res = 1, r = q[0][1];
        for (int i = 1; i < q.size(); i ++ )
            if (q[i][0] >= r) {
                res ++;
                r = q[i][1];
            }
        return q.size() - res;
    }
};
```

##### **Python**

```python
#1. 按照区间右端点从小到大排序（为什么按照区间的右端点排序？ 因为区间的右端点能够覆盖尽可能多的区间） 2. 从左到右选择每个区间；3. 选择方法：能选则选！这样选出来就是最优解
#如果当前区间的左端点大于前一个区间的右端点，说明当前区间可以是一个独立的区间，就保存它；如果当前区间的左端点小于前一个区间的右端点，说明当前区间和前面一个区间重合了，需要删除一个区间，很明显是删除当前区间更好。
#===> 因此我们只需要不断的维护前一个保留区间的右端点即可，只有当前区间的左端点大于前一个保留区间右端点时，我们才会更新保留区间。
#证明贪心解的方法：类似于A=B：1）先证明A>=B ; 2)再证明A<=B ; 3)那就相当于证明了A=B
#那就 证明：贪心解<=最优解(显然) && 贪心解>=最优解 （通过调整法：把最优解调整成贪心解）

class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        #按照右端点排序
        q=sorted(intervals,key=lambda x:x[1])
        if not q:return 0
        res=1;r=q[0][1]
        for i in range(len(q)):
            if q[i][0]>=r:
                res+=1
                r=q[i][1]
        return len(q)-res
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 436. 寻找右区间](https://leetcode-cn.com/problems/find-right-interval/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 二分或堆都可以

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    vector<int> findRightInterval(vector<vector<int>>& intervals) {
        int n = intervals.size();
        // 加入下标
        for (int i = 0; i < n; ++ i ) intervals[i].push_back(i);
        sort(intervals.begin(), intervals.end());
        vector<int> res(n, -1);
        for (auto & intev : intervals) {
            int li = intev[0], ri = intev[1], id = intev[2];
            int l = 0, r = n - 1;
            while (l < r) {
                int m = l + (r - l) / 2;
                if (intervals[m][0] >= ri) r = m;
                else l = m + 1;
            }
            if (intervals[l][0] >= ri) res[id] = intervals[l][2];
        }
        return res;
    }


    typedef tuple<int, int, int> tpi3;
    vector<int> findRightInterval_2(vector<vector<int>>& intervals) {
        vector<tpi3> ve;
        for (int i = 0; i < intervals.size(); ++ i ) ve.push_back({intervals[i][0], intervals[i][1], i});
        sort(ve.begin(), ve.end());

        vector<int> res(intervals.size());
        // 堆 原本想用优先队列 不支持lower_bound 且贪心思路有不足 改set即可
        set<tpi3> q;
        for (int i = ve.size() - 1; i >= 0; -- i ) {
            auto [l, r, id] = ve[i];
            //cout << "l = " << l << " r = " << r << " id = " << id << endl;
            //while (!q.empty() && get<0>(q.top()) < r) q.pop();
            auto it = q.lower_bound({r, INT_MIN, INT_MIN});
            if (it == q.end()) res[id] = -1;
            else res[id] = get<2>(*it);
            q.insert(ve[i]);
        }
        return res;
    }
};
```

##### **Python**

```python
class Solution:
    def findRightInterval(self, nums: List[List[int]]) -> List[int]:
        n=len(nums)
        #排序前给每一项的末尾添加索引
        for i in range(n):
            nums[i].append(i)

        nums.sort(key=lambda x:x[0])
        res=[-1]*n 

        for i in range(n):
            l,r=0,n-1 
            while l<r:
                m=l+(r-l)//2 
                if nums[m][0]<nums[i][1]:
                    l=m+1
                else:r=m 
            if nums[l][0]>=nums[i][1]:
                res[nums[i][2]]=nums[l][2]
        return res
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 632. 最小区间](https://leetcode-cn.com/problems/smallest-range-covering-elements-from-k-lists/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> k 路归并思维

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 显然 答案区间必然是出现在列表中的值
    // K 路归并
    typedef vector<int> VI;
    vector<int> smallestRange(vector<vector<int>>& nums) {
        // 包含所有排列
        priority_queue<VI, vector<VI>, greater<VI>> heap;
        int maxv = INT_MIN;
        for (int i = 0; i < nums.size(); ++ i ) {
            heap.push({nums[i][0], i, 0});
            maxv = max(maxv, nums[i][0]);
        }

        VI res;
        while (heap.size()) {
            auto t = heap.top(); heap.pop();
            int l = t[0], r = maxv;
            if (res.empty() || res[1] - res[0] > r - l)
                res = {l, r};
            int i = t[1], j = t[2] + 1;
            if (j < nums[i].size()) {
                heap.push({nums[i][j], i, j});
                maxv = max(maxv, nums[i][j]);
            } else break;
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

> [!NOTE] **[LeetCode 757. 设置交集大小至少为2](https://leetcode-cn.com/problems/set-intersection-size-at-least-two/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 区间选点的变种 考虑最后两个元素
> 
> 重复

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 区间选点变种
    int intersectionSizeTwo(vector<vector<int>>& intervals) {
        sort(intervals.begin(), intervals.end(), [](vector<int> & a, vector<int> & b) {
            if (a[1] != b[1])
                return a[1] < b[1];
            // 优先考虑比较短的区间 避免重复统计
            return a[0] > b[0];
        });
        
        vector<int> q(1, -1);
        int cnt = 0;
        for (auto & r : intervals)
            // 考虑最后两个元素
            if (r[0] > q[cnt]) {
                q.push_back(r[1] - 1);
                q.push_back(r[1]);
                cnt += 2;
            } else if (r[0] > q[cnt- 1]) {
                q.push_back(r[1]);
                cnt += 1;
            }
        return cnt;
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

> [!NOTE] **[LeetCode 1229. 安排会议日程](https://leetcode-cn.com/problems/meeting-scheduler/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 预选 + 暴力判断
> 
> 可以双指针过 ==> **推导重要**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 预处理暴力**

```cpp
vector<int> minAvailableDuration(vector<vector<int>>& slots1,
                                 vector<vector<int>>& slots2, int duration) {
    vector<vector<int>> s1, s2;
    for (auto& s : slots1)
        if (s[0] + duration <= s[1]) s1.push_back(s);
    for (auto& s : slots2)
        if (s[0] + duration <= s[1]) s2.push_back(s);
    int n1 = s1.size(), n2 = s2.size();
    vector<int> res{INT_MAX, INT_MAX};
    int l, r;
    for (int i = 0; i < n1; ++i)
        for (int j = 0; j < n2; ++j) {
            l = max(s1[i][0], s2[j][0]), r = min(s1[i][1], s2[j][1]);
            if (l + duration <= r && l < res[0])
                res[0] = l, res[1] = l + duration;
        }
    if (res[0] == INT_MAX) return vector<int>{};
    return res;
}
```

##### **C++ 双指针**

```cpp
class Solution {
public:
    vector<int> minAvailableDuration(vector<vector<int>>& slots1, vector<vector<int>>& slots2, int duration) {
        // 排序 保证首先找到的最早
        sort(slots1.begin(), slots1.end());
        sort(slots2.begin(), slots2.end());
        for (int i = 0, j = 0; i < slots1.size() && j < slots2.size(); ) {
            int e1 = slots1[i][1], e2 = slots2[j][1];
            int st = max(slots1[i][0], slots2[j][0]);
            if (st + duration <= min(e1, e2))
                return {st, st + duration};
            // ATTENTION: 重要
            if (e1 < e2)
                i ++ ;
            else
                j ++ ;
        }
        return {};
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

> [!NOTE] **[LeetCode 1235. 规划兼职工作](https://leetcode-cn.com/problems/maximum-profit-in-job-scheduling/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 显然要按照结束时间排序
> 
> 状态 f 有两种定义方式
> 
> 1. i 表示第几个任务，f[i] 表示 i 号之前所有工作能取得的最大收益。则需使用 pre 维护某个任务前面最近结束的任务，计算比较麻烦
> 
> 2. i 表示时间点，f[i] 表示时间点 i 为止的最大收益，随后跑 01 背包 【需要对时间离散化以压缩空间】
> 
> 和 **建筑维修有点类似** 但是又不同 **注意区分**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using PII = pair<int, int>;
    int jobScheduling(vector<int>& startTime, vector<int>& endTime, vector<int>& profit) {
        // Sorting
        vector<PII> ve;
        {
            for (int i = 0; i < profit.size(); ++ i )
                ve.push_back({endTime[i], i});  // endTime & index
            sort(ve.begin(), ve.end());
        }

        // 离散化 Discretization
        vector<int> ts;
        {
            for (auto t : startTime) ts.push_back(t);
            for (auto t : endTime) ts.push_back(t);
            sort(ts.begin(), ts.end());
            ts.erase(unique(ts.begin(), ts.end()), ts.end());
        }

        // DP
        int n = ts.size(), tot = ve.size();
        vector<int> f(n, INT_MIN);
        f[0] = 0;
        for (int i = 1, j = 0; i < n; ++ i ) {
            f[i] = f[i - 1];
            // Thinking: 为相同结束时间的每一个 j 寻找满足 startTIME 的最优答案
            while (j < tot && ve[j].first == ts[i]) {
                int idx = ve[j].second;
                // ts[k] >= startTime[idx]
                // [!] 本质就是离散化后找这个时间所对应的位置
                int k = lower_bound(ts.begin(), ts.end(), startTime[idx]) - ts.begin();
                f[i] = max(f[i], f[k] + profit[idx]);
                j ++ ;
            }
        }
        return f[n - 1];
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

> [!NOTE] **[LeetCode 1751. 最多可以参加的会议数目 II](https://leetcode-cn.com/problems/maximum-number-of-events-that-can-be-attended-ii/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 与上题 [LeetCode 1235. 规划兼职工作](https://leetcode-cn.com/problems/maximum-profit-in-job-scheduling/) 一样
> 
> 但做法不同

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using LL = long long;
    int maxValue(vector<vector<int>>& events, int k) {
        int n = events.size();
        sort(events.begin(), events.end(),
             [](const vector<int> & a, const vector<int> & b) {
                return a[1] < b[1];
        });
        events.insert(events.begin(), vector<int>{0, 0, 0});
        
        vector<vector<LL>> f(n + 1, vector<LL>(k + 1));
        f[0][0] = 0;
        
        for (int i = 1; i <= n; ++ i ) {
            int l = 0, r = i - 1;
            while (l < r) {
                int mid = l + r >> 1;
                if (events[mid + 1][1] >= events[i][0]) r = mid;
                else l = mid + 1;
            }
            
            f[i][0] = 0;
            for (int j = 1; j <= k; ++ j )
                f[i][j] = max(f[i - 1][j], f[l][j - 1] + events[i][2]);
        }
        return f[n][k];
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

> [!NOTE] **[LeetCode 1272. 删除区间](https://leetcode-cn.com/problems/remove-interval/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 模拟 分类讨论即可

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    vector<vector<int>> removeInterval(vector<vector<int>>& intervals,
                                       vector<int>& toBeRemoved) {
        int n = intervals.size(), s = toBeRemoved[0], t = toBeRemoved[1];
        vector<vector<int>> res;
        for (auto p : intervals) {
            if (p[1] < s || p[0] > t) {
                res.push_back(p);
            } else if (p[1] <= t && p[0] >= s) {
                continue;
            } else if (p[1] > t && p[0] < s) {
                res.push_back({p[0], s});
                res.push_back({t, p[1]});
            } else if (p[1] <= t) {
                res.push_back({p[0], s});
            } else if (p[0] >= s) {
                res.push_back({t, p[1]});
            }
        }
        sort(res.begin(), res.end());
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

> [!NOTE] **[LeetCode 1546. 和为目标值的最大数目不重叠非空子数组数目](https://leetcode-cn.com/problems/maximum-number-of-non-overlapping-subarrays-with-sum-equals-target/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 统计区间 随后排序贪心即可

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 某一个前缀和数值出现的下标 要求数量最多 即长度最短 显然可以随时更新
    unordered_map<int, int> m;
    int maxNonOverlapping(vector<int>& nums, int target) {
        int n = nums.size();
        vector<int> sum(n + 1);
        vector<pair<int, int>> ps;
        m[0] = 0;
        for (int i = 1; i <= n; ++i) {
            sum[i] = sum[i - 1] + nums[i - 1];
            if (m.count(sum[i] - target))
                ps.push_back({i, m[sum[i] - target] + 1});
            m[sum[i]] = i;
        }
        // 按结束时间排序
        sort(ps.begin(), ps.end());
        int mx = 0, res = 0;
        for (auto [r, l] : ps)
            if (l <= mx)
                continue;
            else
                ++res, mx = r;
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

### huffman

> [!NOTE] **[AcWing 148. 合并果子](https://www.acwing.com/problem/content/150/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> Huffman 树

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

int n, w;
int main() {
    cin >> n;
    priority_queue<int, vector<int>, greater<int>> heap;
    for (int i = 0; i < n; ++i) {
        cin >> w;
        heap.push(w);
    }
    int res = 0;
    while (heap.size() > 1) {
        int v1 = heap.top();
        heap.pop();
        int v2 = heap.top();
        heap.pop();
        res += v1 + v2;
        heap.push(v1 + v2);
    }
    cout << res << endl;
}
```

##### **Python**

```python
# 每次找最小的两个字 可以用堆（优先队列），小根堆。
# 维护一个最小的heap，每次都取最小的两个，并将“和”加入到heap

if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().split()))

    import heapq

    heapq.heapify(arr)
    res = 0

    while len(arr) >= 2:
        a = heapq.heappop(arr)
        b = heapq.heappop(arr)
        res += a + b
        heapq.heappush(arr, a + b)
    print(res)
```

<!-- tabs:end -->
</details>

<br>

* * *

### 相邻交换

> [!NOTE] **[AcWing 125. 耍杂技的牛](https://www.acwing.com/problem/content/description/127/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 推公式
> 
> 经典相邻交换法贪心

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

int n;
long long w, s;
int main() {
    cin >> n;
    vector<pair<long long, long long>> ve;
    for (int i = 0; i < n; ++i) {
        cin >> w >> s;
        ve.push_back({w + s, s});
    }
    sort(ve.begin(), ve.end());
    long long res = -1e18, sum = 0;
    for (auto [t, s] : ve) {
        sum -= s;
        res = max(res, sum);
        sum += t;
    }
    cout << res << endl;
}
```

##### **Python**

```python
# 算法：按照wi+si从小到大的顺序排，最大的危险系数一定是最小的。
# 证明：1）贪心的答案>=最优解；2）贪心得到的答案<=最优解；

# 证明1）很好证，在这里证明2）：反证法证明。假设最优解不是按照wi+si从小到大排序，那一定存在wi+si>w(i+1)+s(i+1)，此时交换一下i和i+1头牛的位置。
# 交换前：第i头牛：w1+w2+...+w(i-1)-si ; i+1头牛：w1+w2+...+wi-s(i+1)
# 交换后：w1+w2+..+w(i-1)-s(i+1); w1+w2+...+w(i-1)+w(i+1)-si

# 最后会变成：
# s(i+1) --- wi+si 
# si ---- w(i+1)+s(i+1)

# 交换后两个数的最大值 会严格变小。即可证明2）

if __name__ == '__main__':
    n = int(input())
    arr = []
    for _ in range(n):
        arr.append(list(map(int, input().split())))
    arr.sort(key=lambda x: x[0] + x[1])

    res = float('-inf')
    prefix_weight = 0
    for w, s in arr:
        res = max(res, prefix_weight - s)
        prefix_weight += w
    print(res)
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[AcWing 734. 能量石](https://www.acwing.com/problem/content/description/736/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 贪心的假设和推导原则【贪心的微扰（邻项交换）证法】 
> 
> 先贪心排序 国王游戏 耍杂技的牛
> 
> **[如果表示“恰好”，那么需要把所有非法状态初始化成负无穷。]**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <algorithm>
#include <iostream>

using namespace std;

const int N = 110, M = 10010;

int n;
struct Stone {
    int s, e, l;
} stones[N];

// 优先选择 Si/Li 最小的值
bool cmp(Stone a, Stone b) { return a.s * b.l < b.s * a.l; }

int f[N][M];

int main() {
    int T;
    cin >> T;
    for (int C = 1; C <= T; C++) {
        cin >> n;
        int m = 0;
        for (int i = 1; i <= n; i++) {
            int s, e, l;
            cin >> s >> e >> l;
            stones[i] = {s, e, l};
            m += s;
        }

        sort(stones + 1, stones + 1 + n, cmp);

        for (int i = 1; i <= n; i++)
            for (int j = 0; j <= m; j++) {
                f[i][j] = f[i - 1][j];
                if (j >= stones[i].s) {
                    int s = stones[i].s, e = stones[i].e, l = stones[i].l;
                    f[i][j] =
                        max(f[i][j], f[i - 1][j - s] + max(0, e - l * (j - s)));
                }
            }

        int res = 0;
        for (int i = 0; i <= m; i++) res = max(res, f[n][i]);

        printf("Case #%d: %d\n", C, res);
    }

    return 0;
}
```

##### **Python**

```python
#贪心+dp：所有不同吃法的最优解
#1）决定可以选择吃哪些（不吃能量变成0的能量石） 2）按照什么顺序吃能量石 （两维变化，不好直接做）
# 有一个非常巧妙的地方：先做一个贪心，找到最优解的一个小子集里。先证明这个子集的存在，然后考虑这个子集里所有的最大值
# 复习一下贪心题：耍杂技的牛 & 国王游戏

# 贪心思路：发现有可能存在最优解的某些宝石的贡献为00，我们剔除了这些宝石。
# 假设最优解的能量石排列长度为k(1<=k<=n) 因为去掉了那些没有贡献的宝石，位置为：a1,a2,a3…aka1,a2,a3…ak。
# 那么对于任意两个位置i=al,j=al+1(1<=l<k)，在最优解中，交换后两个宝石的贡献总和不会变得更大,(假设之前的总时间为t）：整理后：
# Si∗Lj<=Sj∗Li，调整一下: Si/Li<=Sj/Lj
# 这样，我们只要以如上条件作为宝石间排序的条件，进行一次sortsort。
# 那么最优解的坐标（新的坐标）一定满足：ai<a2<a3<...<ak

# dp: 0/1背包问题，Si 作为费用，max(0,Ei−(t−Si)∗Li) 作为价值 (t为当前花费时长)。
# f[t] 表示当“恰好”花时间t 得到的最大能量


if __name__ == '__main__':
    T = int(input())
    for t in range(T):
        n = int(input())
        nums = []
        m = 0
        for i in range(n):
            nums.append(list(map(int,input().split())))
            m += nums[i][0]
        nums.sort(key = lambda x : x[0] / max(x[2], 1e-10))  # 预处理 排序
        f = [0] * (m + 1)
        for i in range(n):
            s, e, l = nums[i]
            for j in range(m, s - 1, -1):
                f[j] = max(f[j], f[j - s] + e - (j - s) * l)  # 这里 f[j + 1] >= f[j] 不一定成立
        res = f[0]
        for i in range(1, m + 1):   # 这里 f[M] 不一定是最大值，这是因为 j 更大的情况下，(j - s) * l 也就是损耗更大
            res = max(res, f[i])
        print(f'Case #{t + 1}: {res}')
        # print("Case #{}: {}".format(i+1,r[i]))
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 2136. 全部开花的最早一天](https://leetcode-cn.com/problems/earliest-possible-day-of-full-bloom/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 贪心策略推导

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using PII = pair<int, int>;
    int earliestFullBloom(vector<int>& pt, vector<int>& gt) {
        int n = pt.size();
        vector<PII> ve;
        for (int i = 0; i < n; ++ i )
            ve.push_back({gt[i], pt[i]});
        sort(ve.begin(), ve.end());
        reverse(ve.begin(), ve.end());  // 按 gt 降序排序
        int res = 0;
        for (int i = 0, s = 0; i < n; ++ i ) {
            s += ve[i].second;
            res = max(res, s + ve[i].first);
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

### 堆 + 后悔

> [!NOTE] **[Luogu [JSOI2007]建筑抢修](https://www.luogu.com.cn/problem/P4053)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> **贪心思路推导 + 堆**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

// 核心：贪心策略
// 先按 t 贪心，中途再更改
// 1. 按 t 从小到大排序之后，开始轮流遍历每个建筑
// 2. 如果中途某个建筑 i 无法在 t_i 的时间内修复，
//    那么在先前选择修复的建筑中拿出 w_j 最大的 j 号建筑
//    若 w_i < w_j ，则放弃 j 转而修 i。

const int N = 150010;

int n, T;  // T指遍历时经过了多久时间
struct node {
    int w, t;
} a[N];
priority_queue<int> Q;  //优先队列

bool cmp(node x, node y) {
    return x.t < y.t;  //按t从小到大排序
}
int main() {
    scanf("%d", &n);
    for (int i = 1; i <= n; i++)
        scanf("%d%d", &a[i].w, &a[i].t);
    sort(a + 1, a + n + 1, cmp);

    int res = 0;
    for (int i = 1; i <= n; i++)
        //如果无法修复此楼
        if (T + a[i].w > a[i].t) {
            // ai < aj
            if (a[i].w < Q.top()) {
                //注意这里要减掉
                T -= Q.top();
                Q.pop();
                Q.push(a[i].w);
                T += a[i].w;
            }
        } else {
            Q.push(a[i].w);
            res++;
            T += a[i].w;
        }

    printf("%d\n", res);

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

### 贪心策略 堆维护

> [!NOTE] **[LeetCode 407. 接雨水 II](https://leetcode-cn.com/problems/trapping-rain-water-ii/)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> 考虑每个格子最终的高度，该高度减去初始高度就是本格子的水量。
>
> 该高度：从 `[i, j]` 到每个边界的所有路径的最大值的最小值。
>
> 显然无法通过 dp 来转移（状态间的环形依赖），**考虑类似图论的思想**。
>
> -   首先考虑边界，其最终高度显然（为本身）。
> -   向内部扩展时找【最低的，将周围点加入】
>
> **注意正确性推导**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    typedef tuple<int, int, int> tpi3;
    int trapRainWater(vector<vector<int>>& h) {
        if (h.empty() || h[0].empty()) return 0;
        int n = h.size(), m = h[0].size();
        priority_queue<tpi3, vector<tpi3>, greater<tpi3>> heap;
        vector<vector<bool>> st(n, vector<bool>(m));

        for (int i = 0; i < n; ++ i ) {
            st[i][0] = st[i][m - 1] = true;
            heap.push({h[i][0], i, 0});
            heap.push({h[i][m - 1], i, m - 1});
        }
        for (int i = 0; i < m; ++ i ) {
            st[0][i] = st[n - 1][i] = true;
            heap.push({h[0][i], 0, i});
            heap.push({h[n - 1][i], n - 1, i});
        }

        int res = 0;
        vector<int> dx = {-1, 0, 0, 1}, dy = {0, -1, 1, 0};
        while (!heap.empty()) {
            auto [nh, x, y] = heap.top();
            heap.pop();
            res += nh - h[x][y];
            for (int i = 0; i < 4; ++ i ) {
                int nx = x + dx[i], ny = y + dy[i];
                if (nx < 0 || nx >= n || ny < 0 || ny >= m || st[nx][ny]) continue;
                heap.push({max(h[nx][ny], nh), nx, ny});
                st[nx][ny] = true;
            }
        }
        return res;
    }
};
```

##### **C++ yxc**

```cpp
// yxc
class Solution {
public:
    struct Cell {
        int h, x, y;
        bool operator< (const Cell& t) const {
            return h > t.h;
        }
    };

    int trapRainWater(vector<vector<int>>& h) { 
        if (h.empty() || h[0].empty()) return 0;
        int n = h.size(), m = h[0].size();
        priority_queue<Cell> heap;
        vector<vector<bool>> st(n, vector<bool>(m));
        for (int i = 0; i < n; i ++ ) {
            st[i][0] = st[i][m - 1] = true;
            heap.push({h[i][0], i, 0});
            heap.push({h[i][m - 1], i, m - 1});
        }
        for (int i = 1; i + 1 < m; i ++ ) {
            st[0][i] = st[n - 1][i] = true;
            heap.push({h[0][i], 0, i});
            heap.push({h[n - 1][i], n - 1, i});
        }
        int res = 0;
        int dx[] = {-1, 0, 1, 0}, dy[] = {0, 1, 0, -1};
        while (heap.size()) {
            auto t = heap.top();
            heap.pop();
            res += t.h - h[t.x][t.y];

            for (int i = 0; i < 4; i ++ ) {
                int x = t.x + dx[i], y = t.y + dy[i];
                if (x >= 0 && x < n && y >= 0 && y < m && !st[x][y]) {
                    heap.push({max(h[x][y], t.h), x, y});
                    st[x][y] = true;
                }
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

> [!NOTE] **[LeetCode 502. IPO](https://leetcode-cn.com/problems/ipo/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 易知贪心规则
> 
> 记录实现

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 优先选择当前资本下可选 且利润最大的项目
    // 细化：
    //    首先将项目按照资本从小到大排序。
    //    维护一个大根堆，根据当前资本 W，将可以开展的项目的利润放到堆中。
    //    每开始一个新项目时，从堆中取最大利润的项目，做完后增加 W，接着维护堆。
    int findMaximizedCapital(int k, int W, vector<int>& Profits, vector<int>& Capital) {
        vector<pair<int, int>> q;
        int n = Profits.size();
        for (int i = 0; i < n; ++ i ) q.push_back({Capital[i], Profits[i]});
        sort(q.begin(), q.end());

        priority_queue<int> heap;
        int res = 0, i = 0;
        while (k -- ) {
            while (i < n && q[i].first <= W) heap.push(q[i].second), ++ i ;
            if (heap.empty()) break;
            auto t = heap.top(); heap.pop();
            W += t;
        }
        return W;
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

> [!NOTE] **[LeetCode 630. 课程表 III](https://leetcode-cn.com/problems/course-schedule-iii/)**
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
    int scheduleCourse(vector<vector<int>>& courses) {
        // 按结束时间排序
        sort(courses.begin(), courses.end(), [](vector<int> & a, vector<int> & b) {
            return a[1] < b[1];
        });
        priority_queue<int> heap;
        int tot = 0;
        for (auto & c : courses) {
            tot += c[0];
            heap.push(c[0]);
            if (tot > c[1]) {
                // 去除最大值
                tot -= heap.top();
                heap.pop();
            }
        }
        return heap.size();
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

> [!NOTE] **[LeetCode 659. 分割数组为连续子序列](https://leetcode-cn.com/problems/split-array-into-consecutive-subsequences/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 堆模拟 以及非常好的贪心
> 
> 贪心的trick思维 转化为连续组
> 
> 重复理解 **注意要求：升序数组 heap也要理解**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 堆**

```cpp
class Solution {
public:
    typedef pair<int, int> PII;
    bool isPossible(vector<int>& nums) {
        int n = nums.size();
        priority_queue<PII, vector<PII>, greater<PII>> pq;
        for (auto v : nums) {
            while (!pq.empty() && pq.top().first < v - 1) {
                if (pq.top().second < 3) return false;
                pq.pop();
            }
            if (pq.empty() || pq.top().first >= v) pq.push({v, 1});
            else pq.push({v, pq.top().second + 1}), pq.pop();
        }
        while (!pq.empty()) {
            if (pq.top().second < 3) return false;
            pq.pop();
        }
        return true;
    }
};
```

##### **C++ trick**

```cpp
class Solution {
public:
    bool isPossible(vector<int>& nums) {
        unordered_map<int, int> cnt1, cnt2;
        for (auto x: nums) cnt1[x] ++ ;
        for (auto x: nums) {
            if (!cnt1[x]) continue;
            if (cnt2[x - 1]) {
                cnt2[x - 1] -- ;
                cnt2[x] ++ ;
                cnt1[x] -- ;
            } else if (cnt1[x + 1] && cnt1[x + 2]) {
                cnt2[x + 2] ++ ;
                cnt1[x] --, cnt1[x + 1] --, cnt1[x + 2] -- ;
            } else return false;
        }
        return true;
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

> [!NOTE] **[LeetCode 1353. 最多可以参加的会议数目](https://leetcode-cn.com/problems/maximum-number-of-events-that-can-be-attended/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 线性扫描问题 考虑每一天可选的会议范围 并选择结束时间最早的会议
> 
> 主要是使用优先队列维护 不能简单贪心

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
const int MAX = 1e5 + 1;
class Solution {
public:
    int maxEvents(vector<vector<int>>& events) {
        vector<vector<int>> left(MAX);
        for (int i = 0; i < events.size(); ++i)
            left[events[i][0]].emplace_back(i);
        // 每个时间点开始的会议 存下标

        int ans = 0;
        priority_queue<int, vector<int>, greater<>> pq;
        for (int i = 1; i < MAX; ++i) {
            for (int j : left[i]) pq.push(events[j][1]);
            while (!pq.empty() && pq.top() < i) pq.pop();
            if (!pq.empty()) {
                pq.pop();
                ans++;
            }
        }
        return ans;
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

> [!NOTE] **[LeetCode 1383. 最大的团队表现值](https://leetcode-cn.com/problems/maximum-performance-of-a-team/)**
> 
> 题意: 
> 
> 两个数组 speed 和 efficiency ，其中 speed[i] 和 efficiency[i] 分别代表第 i 位工程师的速度和效率。
> 
> 请你返回由最多 k 个工程师组成的 最大团队表现值。
> 
> 团队表现值 的定义为：一个团队中「所有工程师速度的和」乘以他们「效率值中的最小值」。
> 
> 取模

> [!TIP] **思路**
> 
> 按照效率进行降序排序，每个人作为最低效率时，在其左侧找出至多K - 1个最大速度即可(再加上这个人的速度组成K个)

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int maxPerformance(int n, vector<int>& speed, vector<int>& efficiency,
                       int k) {
        vector<pair<int, int>> ns;
        for (int i = 0; i < n; ++i)
            ns.push_back({speed[i], efficiency[i]});
        sort(ns.begin(), ns.end(),
             [](const pair<int, int>& a, const pair<int, int>& b) -> bool {
                 return a.second > b.second;
             });
        long long res = 0, mod = 1e9 + 7, sum = 0;
        priority_queue<int, vector<int>, greater<int>> pq;
        for (int i = 0; i < n; ++i) {
            pq.push(ns[i].first);
            sum += ns[i].first;
            if (pq.size() > k) {
                sum -= pq.top();
                pq.pop();
            }
            res = max(res, (long long)sum * ns[i].second);
        }
        return res % mod;
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

> [!NOTE] **[LeetCode 1488. 避免洪水泛滥](https://leetcode-cn.com/problems/avoid-flood-in-the-city/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 贪心 写法思路完全正确 vector二分查找超时
> 
> 数组改成set就过了

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ vector TLE**

```cpp
// 最后一个用例超时
class Solution {
public:
    // 某一个湖泊 上一次下雨的下标（是哪一天） 无数个湖泊所以用map
    unordered_map<int, int> m;
    vector<int> avoidFlood(vector<int>& rains) {
        int n = rains.size();
        // 没下雨的天下标
        vector<int> zero;
        vector<int> res;
        for (int i = 0; i < n; ++i) {
            if (rains[i]) {
                if (m[rains[i]]) {
                    auto it =
                        lower_bound(zero.begin(), zero.end(), m[rains[i]]);
                    if (it == zero.end())
                        return vector<int>{};
                    else {
                        res[*it - 1] = rains[i];
                        zero.erase(it);
                    }
                }
                m[rains[i]] = i + 1;
                res.push_back(-1);
            } else {
                // rains[i] == 0
                zero.push_back(i + 1);
                res.push_back(0);  // 默认什么都不动
            }
        }
        for (int i = 0; i < n; ++i) {
            if (!res[i]) res[i] = 1;
        }
        return res;
    }
};
```

##### **C++ set AC**

```cpp

class Solution {
public:
    // 某一个湖泊 上一次下雨的下标（是哪一天） 无数个湖泊所以用map
    unordered_map<int, int> m;
    vector<int> avoidFlood(vector<int>& rains) {
        int n = rains.size();
        // 没下雨的天下标
        set<int> zero;
        // vector<int> res(n,1);
        vector<int> res;
        for (int i = 0; i < n; ++i) {
            if (rains[i]) {
                if (m[rains[i]]) {
                    auto it = zero.lower_bound(m[rains[i]]);
                    if (it == zero.end())
                        return vector<int>{};
                    else {
                        res[*it - 1] = rains[i];
                        zero.erase(it);
                    }
                }
                m[rains[i]] = i + 1;
                // res[i] = -1;
                res.push_back(-1);
            } else {
                // rains[i] == 0
                zero.insert(i + 1);
                res.push_back(1);
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

> [!NOTE] **[LeetCode 1606. 找到处理最多请求的服务器](https://leetcode-cn.com/problems/find-servers-that-handled-most-number-of-requests/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
>
> 核心在**构造待选列表以降低复杂度** 。
>
> 遍历每一个 `arrival` ， 数据范围较大不能每次遍历过程都模拟向后寻找。
>
> 考虑使用优先队列维护每一个服务器的结束时间，在每一个 `arrival` 到来的处理过程中，先更新之前曾执行任务的服务器列表 `busy` ，如果比当前时间小则加入可用集合 `svr` 。处理结束后，所有 `busy` 中的服务器均在当前处于忙状态。若 `svr` 空则无可用服务器，否则按题意找下标服务器，从 `svr` 清除并加入 `busy` 。

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    vector<int> busiestServers(int k, vector<int>& arrival, vector<int>& load) {
        int n = arrival.size(), mx = 0;
        vector<int> cnt(n);
        set<int> svr;
        for (int i = 0; i < k; ++i) svr.insert(i);
        priority_queue<pair<int, int>> busy;
        for (int i = 0; i < n; ++i) {
            int p = i % k, t = arrival[i], c = load[i];
            while (!busy.empty() && busy.top().first * -1 <= t) {
                svr.insert(busy.top().second);
                busy.pop();
            }
            if (svr.empty()) continue;
            auto it = svr.lower_bound(p);
            if (it == svr.end()) it = svr.begin();
            p = *it;
            svr.erase(p);
            busy.push({-t - c, p});
            mx = max(mx, ++cnt[p]);
        }
        vector<int> res;
        for (int i = 0; i < n; ++i)
            if (cnt[i] == mx) res.push_back(i);
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

> [!NOTE] **[LeetCode 1834. 单线程 CPU](https://leetcode-cn.com/problems/single-threaded-cpu/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 外层**写法**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using LL = long long;
    struct Task {
        int et, pt, id;
        // heap
        bool operator < (const Task & t) const {
            return pt == t.pt ? id > t.id : pt > t.pt;
        }
    };
    
    vector<int> getOrder(vector<vector<int>>& tasks) {
        int n = tasks.size();
        vector<Task> qt;
        for (int i = 0; i < n; ++ i )
            qt.push_back({tasks[i][0], tasks[i][1], i});
        sort(qt.begin(), qt.end(), [](const Task & a, const Task & b) {
            return a.et == b.et ? a.pt < b.pt : a.et < b.et;
        });
        
        priority_queue<Task> heap;
        vector<int> res;
        int last = 0;
        LL time = 0;
        // 每次过程执行一个任务
        while (heap.size() || last < n) {
            // 细节 更新time基准【所谓的时间戳快进】
            if (heap.empty())
                time = max(time, (LL)qt[last].et);
            while (last < n && qt[last].et <= time)
                heap.push(qt[last ++ ]);
            
            auto t = heap.top(); heap.pop();
            res.push_back(t.id);
            time += t.pt;
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

> [!NOTE] **[LeetCode 1942. 最小未被占据椅子的编号](https://leetcode-cn.com/problems/the-number-of-the-smallest-unoccupied-chair/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 维护堆即可 略
> 
> 注意 `当时间 i 为 st 时仍需弹出 used` ，或者直接加个 break 即可
> 
> **two-pointer trick**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using PII = pair<int, int>;
    
    int smallestChair(vector<vector<int>>& times, int targetFriend) {
        int n = times.size();
        int st = times[targetFriend][0];
        sort(times.begin(), times.end());
        
        priority_queue<int, vector<int>, greater<int>> q;
        for (int i = 0; i < n; ++ i )
            q.push(i);
        priority_queue<PII, vector<PII>, greater<PII>> used;
        
        for (int i = 0, j = 0; i <= st; ++ i ) {
            while (used.size() && used.top().first <= i) {
                int id = used.top().second;
                used.pop();
                q.push(id);
            }
            // ATTENTION
            if (i == st)
                break;
            if (i == times[j][0]) {
                int id = q.top(); q.pop();
                used.push({times[j][1], id});
                j ++ ;
            }
        }
        return q.top();
    }
};
```

##### **C++ 直接扫完**

或者直接扫完返回结果

```cpp
class Solution {
public:
    int smallestChair(vector<vector<int>>& times, int targetFriend) {
        vector<vector<int>> v(100010);
        vector<vector<int>> w(100010);
        int n = times.size();
        for (int i = 0; i < n; ++i) {
            int x = times[i][0], y = times[i][1];
            v[x].push_back(i);
            w[y].push_back(i);
        }
        set<int> H;
        for (int i = 0; i < n; ++i) H.insert(i);
        vector<int> res(n);
        for (int i = 1; i <= 100000; ++i) {
            for (auto id : w[i]) {
                H.insert(res[id]);
            }
            for (auto id : v[i]) {
                res[id] = *H.begin();
                H.erase(res[id]);
            }
        }
        return res[targetFriend];
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

> [!NOTE] **[Luogu 最大乘积](https://www.luogu.com.cn/problem/P1249)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 贪心 + 大数

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

// 1. 贪心从 2 开始递增累加(s) 直到大于 n ，把 s-n 删掉
// 2. 高精度累乘

vector<int> mul(vector<int> & a, int b) {
    vector<int> c;
    for (int i = 0, t = 0; i < a.size() || t; ++ i ) {
        if (i < a.size())
            t += a[i] * b;
        c.push_back(t % 10);
        t /= 10;
    }
    while (c.size() > 1 && c.back() == 0)
        c.pop_back();
    return c;
}

int main() {
    int n;
    cin >> n;

    int v = 1, s = 0;
    vector<int> ve;
    while (s < n) {
        v ++ ;
        ve.push_back(v);
        
        s += v;
    }
    
    int nouse = 0;
    if (s > n)
        nouse = s - n;
    
    vector<int> res(1, 1);
    for (auto v : ve) {
        if (v == nouse)
            continue;
        cout << v << ' ';
        res = mul(res, v);
    }
    cout << endl;

    for (int i = res.size() - 1; i >= 0; -- i )
        cout << res[i];
    cout << endl;
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

> [!NOTE] **[Luogu [AHOI2018初中组]分组](https://www.luogu.com.cn/problem/P4447)**
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

using PII = pair<int, int>;
#define x first
#define y second
const int N = 1e5 + 10;

int n;
unordered_map<int, int> cnt;
vector<PII> ve;

int main() {
    cin >> n;
    for (int i = 0; i < n; ++ i ) {
        int v;
        cin >> v;
        cnt[v] ++ ;
    }
    
    for (auto [k, v] : cnt)
        ve.push_back({k, v});
    sort(ve.begin(), ve.end());
    
    int res = 2e9;
    int m = ve.size();
    for (int i = 0; i < m; ++ i ) {
        // 先消耗一个连续上升区间
        ve[i].y -- ;
        int j = i + 1;
        while (j < m && ve[j].x == ve[j - 1].x + 1 && ve[j].y > ve[j - 1].y)
            ve[j].y -- , j ++ ;
        
        // [l, j - 1]
        int len = j - i;
        if (len < res)
            res = len;

        int k = i;
        while (k < m && ve[k].y == 0)
            k ++ ;
        i = k - 1;
    }
    cout << res << endl;
    
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

> [!NOTE] **[LeetCode 134. 加油站](https://leetcode-cn.com/problems/gas-station/)**
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
// 最优
class Solution {
public:
    int canCompleteCircuit(vector<int>& gas, vector<int>& cost) {
        int n = gas.size();
        int tot = 0, cur = 0, s = 0;
        for (int i = 0; i < n; ++ i ) {
            tot += gas[i] - cost[i];
            cur += gas[i] - cost[i];
            if (cur < 0) {
                s = i + 1;
                cur = 0;
            }
        }
        return tot < 0 ? -1 : s;
    }
};
```

##### **C++ 2**

```cpp
class Solution {
public:
    int canCompleteCircuit(vector<int>& gas, vector<int>& cost) {
        int n = gas.size();
        for (int i = 0, j; i < n; ) {  // 枚举起点
            int left = 0;
            for (j = 0; j < n; j ++ ) {
                int k = (i + j) % n;
                left += gas[k] - cost[k];
                if (left < 0) break;
            }
            if (j == n) return i;
            i = i + j + 1;
        }

        return -1;
    }
};
```

##### **Python**

```python
# 先枚举，然后优化。
# 枚举所有起点，去循环判断一边，有没有出现负数的情况。
# 在i个点，判断在当前点加的油量，能不能走到第i+1个站，依次循环向下走。如果走到第j个战后，没有办法走到j+1个站，说明第i个点开始走，不合法。
# 那就开始遍历其他点作为起点。贪心的部分就是：如果第i个站 不合法，那第【i+1,j】这些站点 都不可能作为起始站点。
# 所以每个站 最多只能被遍历一遍。

class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        n = len(gas)
        for i in range(n): # 枚举起点
            left = 0 
            for j in range(n):
                k = (i + j) % n 
                left += gas[k] - cost[k]
                if left < 0:
                    break 
            if left >= 0:return i
            i = i + j + 1
        return -1
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 330. 按要求补齐数组](https://leetcode-cn.com/problems/patching-array/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 1. 假设当前我们能生成的数字为[0, x)，如果 nums[i] 存在小于等于 x 的数，那么我们先选择添加这个数并将我们的 x = x + nums[i] 。
> 
> 2. 如果不存在的话，那么说明我们需要自己添加一个新数了，我们选择加入 x ,所以 x = x + x ，同时记录答案。

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int minPatches(vector<int>& nums, int n) {
        long long x = 1;
        int i = 0, res = 0;
        while (x <= n) {
            if (i < nums.size() && nums[i] <= x) x += nums[i ++ ];
            else x += x, ++ res;
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

> [!NOTE] **[LeetCode 376. 摆动序列](https://leetcode-cn.com/problems/wiggle-subsequence/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 经典 dp OR 贪心
> 
> 另yxc贪心策略的证明

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// 1. 朴素
class Solution {
public:
    int wiggleMaxLength(vector<int>& nums) {
        int n = nums.size();
        vector<int> up(n + 1), dn(n + 1);
        int res = 0;
        for (int i = 1; i <= n; ++ i ) {
            up[i] = dn[i] = 1;
            for (int j = 1; j < i; ++ j ) {
                //cout << i << " " << j << endl;
                if (nums[i - 1] > nums[j - 1]) up[i] = max(up[i], dn[j] + 1);
                if (nums[i - 1] < nums[j - 1]) dn[i] = max(dn[i], up[j] + 1);
            }
            res = max(res, max(up[i], dn[i]));
        }
        return res;
    }
};

// 2. 优化
class Solution {
public:
    int wiggleMaxLength(vector<int>& nums) {
        int n = nums.size();
        if (n < 2) return n;
        vector<int> up(n + 1), dn(n + 1);
        up[1] = dn[1] = 1;
        for (int i = 2; i <= n; ++ i ) {
            if (nums[i - 1] > nums[i - 2]) {
                up[i] = dn[i - 1] + 1;
                dn[i] = dn[i - 1];
            } else if (nums[i - 1] < nums[i - 2]) {
                up[i] = up[i - 1];
                dn[i] = up[i - 1] + 1;
            } else {
                up[i] = up[i - 1];
                dn[i] = dn[i - 1];
            }
        }
        return max(up[n], dn[n]);
    }
};

// 3. 贪心
class Solution {
public:
    int wiggleMaxLength(vector<int>& nums) {
        int n = nums.size();
        if (n < 2) return n;
        // type > 0 代表上升
        int res = 1, type = 0;
        for (int i = 2; i <= n; ++ i ) {
            if (nums[i - 1] > nums[i - 2] && type <= 0) {
                ++ res;
                type = 1;
            } else if (nums[i - 1] < nums[i - 2] && type >= 0) {
                ++ res;
                type = -1;
            }
        }
        return res;
    }
};
```

##### **C++ yxc**

```cpp
// yxc
// 取两端点以及中间的每一个极值（证明）
class Solution {
public:
    int wiggleMaxLength(vector<int>& nums) {
        nums.erase(unique(nums.begin(), nums.end()), nums.end());
        if (nums.size() <= 2) return nums.size();
        int res = 2;
        for (int i = 1; i + 1 < nums.size(); i ++ ) {
            int a = nums[i - 1], b = nums[i], c = nums[i + 1];
            if (b > a && b > c || b < a && b < c) res ++ ;
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

> [!NOTE] **[LeetCode 1405. 最长快乐字符串](https://leetcode-cn.com/problems/longest-happy-string/)**
> 
> 题意: 
> 
> 生成任意一个尽可能长的快乐字符串(不包含3个连续相同字母) abc各自至多使用入参'a''b''c'次

> [!TIP] **思路**
> 
> 贪心 选择每次和前面不重复的剩下最多的 依据剩下的数量选择放1/2个
> 
> TODO: 更优雅的实现

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    string longestDiverseString(int a, int b, int c) {
        string s;
        int last = 0;
        while (a || b || c) {
            int maxv = 0, newlast = 0, mindif = INT_MAX;
            if (last != 1 && maxv < a) {
                maxv = a;
                newlast = 1;
            }
            if (last != 2 && maxv < b) {
                maxv = b;
                newlast = 2;
            }
            if (last != 3 && maxv < c) {
                maxv = c;
                newlast = 3;
            }
            // 前面选出abc中剩下可选最多的
            if (newlast == 0) break;
            if (newlast == 1) {
                if (maxv > 1 && a >= b + 2 && a >= c + 2) {
                    s += "aa";
                    a -= 2;
                } else {
                    s += "a";
                    a -= 1;
                }
            } else if (newlast == 2) {
                if (maxv > 1 && b >= a + 2 && b >= c + 2) {
                    s += "bb";
                    b -= 2;
                } else {
                    s += "b";
                    b -= 1;
                }
            } else if (newlast == 3) {
                if (maxv > 1 && c >= a + 2 && c >= b + 2) {
                    s += "cc";
                    c -= 2;
                } else {
                    s += "c";
                    c -= 1;
                }
            }
            last = newlast;
        }
        return s;
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

> [!NOTE] **[LeetCode 1520. 最多的不重叠子字符串](https://leetcode-cn.com/problems/maximum-number-of-non-overlapping-substrings/)** [TAG]
> 
> 题意: 
> 
> 给你一个只包含小写字母的字符串 s ，你需要找到 s 中最多数目的非空子字符串，满足如下条件：
> 
> 这些字符串之间互不重叠，也就是说对于任意两个子字符串 s[i..j] 和 s[k..l] ，要么 j < k 要么 i > l 。
> 
> 如果一个子字符串包含字符 c ，那么 s 中所有 c 字符都应该在这个子字符串中。

> [!TIP] **思路**
> 
> 贪心
> 
> 先获取满足条件2的所有子串，再依据条件1调整子串列表，随后按照子串长度贪心旋转即可

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    vector<string> maxNumOfSubstrings(string s) {
        int n = s.size(), inf = 1e9;
        vector<vector<int>> pos(26);
        vector<int> l(26, inf), r(26, -inf);
        for (int i = 0; i < n; ++i) {
            int v = s[i] - 'a';
            pos[v].push_back(i);
            l[v] = min(l[v], i);
            r[v] = max(r[v], i);
        }
        // 扩边界
        vector<pair<int, int>> H;
        for (int i = 0; i < 26; ++i)
            if (l[i] != inf) {
                int nl = l[i], nr = r[i];
                // ===> for(int j = l[i]+1; j < r[i]; ++j) {
                // 思考 应该是下面这样
                for (int j = l[i] + 1; j < nr; ++j) {
                    int v = s[j] - 'a';
                    if (l[v] >= nl && r[v] <= nr) continue;
                    nl = min(nl, l[v]);
                    nr = max(nr, r[v]);
                    j = nl;  // 前面 continue确保此处一定会更新
                }
                H.push_back({nl, nr});
            }
        // 排序
        sort(H.begin(), H.end(),
             [&](const pair<int, int>& p1, const pair<int, int>& p2) {
                 return p1.second - p1.first < p2.second - p2.first;
             });
        vector<string> res;
        // 排除内包含
        vector<bool> vis(H.size());
        for (int i = 0; i < H.size(); ++i) {
            if (vis[i]) continue;
            auto [x, y] = H[i];
            for (int j = 0; j < H.size(); ++j) {
                auto [lj, rj] = H[j];
                if (lj <= x && y <= rj) vis[j] = true;
            }
            res.push_back(s.substr(x, y - x + 1));
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

> [!NOTE] **[LeetCode 1953. 你可以工作的最大周数](https://leetcode-cn.com/problems/maximum-number-of-weeks-for-which-you-can-work/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 贪心加思维
> 
> 考虑某个最长的任务以及剩下其他所有任务之和
> 
> 1. 如果其他任务之和小于最长任务，可完成周数为其他任务之和的二倍加一
> 2. 如果其他任务之和大于等于最长任务，则所有任务皆可完成，直接累加
> 
> 做的太慢了。。。

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using LL = long long;
    long long numberOfWeeks(vector<int>& ms) {
        sort(ms.begin(), ms.end());
        int n = ms.size();
        LL res = 0;
        for (int i = 0; i < n - 1; ++ i )
            res += (LL)ms[i];
        if (res < ms[n - 1])
            res += res + 1;
        else
            res += ms[n - 1];
        return res;
    }
};
```

##### **C++ 直接遍历**

也可以直接遍历同时取最值：

```cpp
using LL = long long;
class Solution {
public:
    long long numberOfWeeks(vector<int>& milestones) {
        LL sum = 0, m = 0;
        for (LL x : milestones)
            sum += x, m = max(m, x);
        if (m > sum - m + 1)
            return (sum - m) * 2 + 1;
        return sum;
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

> [!NOTE] **[LeetCode 2071. 你可以安排的最多任务数目](https://leetcode-cn.com/problems/maximum-number-of-tasks-you-can-assign/)** TAG
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 细节贪心
> 
> 另有 **单调队列** 解法

<details>
<summary>详细代码</summary>
<!-- tabs:start -->


##### **C++ 单调队列 最优**

```cpp
class Solution {
public:
    // [v, has_use_pill, id] , ATTENTION: has_use_pill should be takon account
    using TIII = tuple<int, int, int>;
    const static int N = 5e4 + 10;
    
    int n, m, ps, st;
    vector<int> ts, ws;
    int q[N];
    
    // https://leetcode-cn.com/problems/maximum-number-of-tasks-you-can-assign/solution/c-er-fen-tao-tan-xin-by-blackskygg-kmbu/
    bool check(int mid) {
        // 维护 [..., v + st] 的队列，当前worker的最优解一定是选择第一个or最后一个 或false
        int hh = 0, tt = -1, pills = ps;
        for (int i = max(m - mid, 0), j = 0; i < m; ++ i ) {
            int v = ws[i];
            while (j < mid && ts[j] <= v + st)
                q[ ++ tt] = j ++;
            // 为空 当前worker找不到可以解决的task
            if (hh > tt)
                return false;
            if (ts[q[hh]] <= v)
                hh ++ ;
            else {
                // 吃药 解决一个可解决的最大的
                if (pills)
                    tt -- , pills -- ;
                else
                    return false;
            }
        }
        return true;
    }
    
    int maxTaskAssign(vector<int>& tasks, vector<int>& workers, int pills, int strength) {
        this->ts = tasks, this->ws = workers;
        sort(ts.begin(), ts.end());
        sort(ws.begin(), ws.end());
        n = ts.size(), m = ws.size();
        this->ps = pills, this->st = strength;
        
        int l = 1, r = min(n, m) + 1;
        while (l < r) {
            int mid = l + (r - l) / 2;
            if (check(mid))
                l = mid + 1;
            else
                r = mid;
        }
        return l - 1;
    }
};
```

##### **C++ 二分+二分贪心**

```cpp

class Solution {
public:
    // [v, has_use_pill, id] , ATTENTION: has_use_pill should be takon account
    using TIII = tuple<int, int, int>;
    
    int n, m, ps, st;
    vector<int> ts, ws;
    
    bool check(int mid) {
        multiset<int> S;
        for (int i = max(m - mid, 0); i < m; ++ i )
            S.insert(ws[i]);
        
        vector<int> ve;
        for (int i = 0; i < mid && i < n; ++ i )
            ve.push_back(ts[i]);
        
        // 从大到小考虑
        int pills = ps;
        for (int i = ve.size() - 1; i >= 0; -- i ) {
            auto it = S.lower_bound(ve[i]);
            if (it != S.end())
                S.erase(it);
            else {
                // ATTENTION: 找到一个吃药后可以解决当前任务的最小的一个
                it = S.lower_bound(ve[i] - st);
                if (it != S.end() && pills) {
                    S.erase(it), pills -- ;
                } else
                    return false;
            }
        }
        return true;
    }
    
    int maxTaskAssign(vector<int>& tasks, vector<int>& workers, int pills, int strength) {
        this->ts = tasks, this->ws = workers;
        sort(ts.begin(), ts.end());
        sort(ws.begin(), ws.end());
        n = ts.size(), m = ws.size();
        this->ps = pills, this->st = strength;
        
        int l = 1, r = min(n, m) + 1;
        while (l < r) {
            int mid = l + (r - l) / 2;
            if (check(mid))
                l = mid + 1;
            else
                r = mid;
        }
        return l - 1;
    }
};
```

##### **C++ WA**

显然容易想到二分并选择最小的 mid 个任务与最强的 mid 个工人。但二分策略有问题（TODO），比赛时的 WA 代码：

错误原因：**显然并非需要尽着最小的去吃药**

```cpp
// 46 / 48 个通过测试用例
class Solution {
public:
    // [v, has_use_pill, id] , ATTENTION: has_use_pill should be takon account
    using TIII = tuple<int, int, int>;
    
    int n, m, ps, st;
    vector<int> ts, ws;
    
    bool check(int mid) {
        priority_queue<TIII, vector<TIII>, greater<TIII>> heap;
        for (int i = max(m - mid, 0); i < m; ++ i )
            heap.push({ws[i], i, 0}), heap.push({ws[i] + st, i, 1});
        
        vector<int> ve;
        for (int i = 0; i < mid && i < n; ++ i )
            ve.push_back(ts[i]);
        
        int p = 0, pills = ps;
        unordered_set<int> hash;
        while (heap.size() && p < mid) {
            auto [v, id, used] = heap.top(); heap.pop();
            if (v < ve[p] || hash.count(id))
                continue;
            
            if (used && pills > 0) {
                hash.insert(id);
                p ++ ;
                pills -- ;
            } else if (!used) {
                hash.insert(id);
                p ++ ;
            }
        }
        return p >= mid;
    }
    
    int maxTaskAssign(vector<int>& tasks, vector<int>& workers, int pills, int strength) {
        this->ts = tasks, this->ws = workers;
        sort(ts.begin(), ts.end());
        sort(ws.begin(), ws.end());
        n = ts.size(), m = ws.size();
        this->ps = pills, this->st = strength;
        
        int l = 1, r = min(n, m) + 1;
        while (l < r) {
            int mid = l + (r - l) / 2;
            if (check(mid))
                l = mid + 1;
            else
                r = mid;
        }
        return l - 1;
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

> [!NOTE] **[LeetCode 2193. 得到回文串的最少操作次数](https://leetcode-cn.com/problems/minimum-number-of-moves-to-make-palindrome/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 复杂贪心
> 
> 数据范围更大的同样题目: https://www.luogu.com.cn/problem/P5041
> 
> 关于回文串的一些重要特性：ref **[Codeforces C. Palindrome Transformation](https://codeforces.com/problemset/problem/486/C)**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ TODO**

```cpp
class Solution {
public:
    int minMovesToMakePalindrome(string s) {
        // 记录下标
        vector<int> p[26];
        for (int i = 0; i < n; ++ i )
            p[s[i] - 'a'].push_back(i);
        
        // 生成反串以及对应的在原串中的字符下标 ==> TODO
        // 原理 ref: 
        int cnt[26] = {};
        auto t = s; reverse(t.begin(), t.end());
        vector<int> ve(n);
        for (int i = 0; i < n; ++ i )
            ve[i] = p[t[i] - 'a'][c[t[i] - 'a'] ++ ];
        
        int res = 0;
        for (int i = 0; i < n; ++ i )
            for (int j = 0; j < i; ++ j )
                if (a[j] > a[i])
                    res ++ ;
        return res / 2; // ==> TODO
    }
};
```

##### **C++ 标准贪心**

```cpp
class Solution {
public:
    // 贪心的思想：
    // 每一轮对于最左侧的字符 α，找到最大的下标 j，满足 sj=α，将 sj 移动到最右侧，然后同时去掉最左侧和最右侧的字符。
    // 如果找不到 jj，由于题意保证可以得到回文串，则说明 alpha 在当前字符串中仅出现了一次，需要放到最中间的位置上。
    // TODO: 可信证明
    int minMovesToMakePalindrome(string s) {
        int n = s.size(), res = 0;
        for (int i = 0; i < n; ++ i ) {
            int p = -1;
            for (int j = n - 1; j > i; -- j )
                if (s[j] == s[i]) {
                    p = j;
                    break;
                }
            
            if (p == -1) {
                // 奇数个 移动到中间
                res += s.size() / 2 - i;
                // 右边界不变 直接continue
                continue;
            }
            
            for (int j = p; j < n - 1; ++ j )
                s[j] = s[j + 1], res ++ ;
            n -- ;
        }
        return res;
    }
};
```

##### **C++ 贪心 + bit优化**

```cpp
class Solution {
public:
    // 贪心的思想：
    // 每一轮对于最左侧的字符 α，找到最大的下标 j，满足 sj=α，将 sj 移动到最右侧，然后同时去掉最左侧和最右侧的字符。
    // 如果找不到 jj，由于题意保证可以得到回文串，则说明 alpha 在当前字符串中仅出现了一次，需要放到最中间的位置上。
    // TODO: 可信证明
    const static int N = 2e3 + 10;
    
    int tr[N];
    void init() {
        memset(tr, 0, sizeof tr);
    }
    int lowbit(int x) {
        return x & -x;
    }
    void add(int x, int c) {
        for (int i = x; i < N; i += lowbit(i))
            tr[i] += c;
    }
    int query(int x) {
        int ret = 0;
        for (int i = x; i; i -= lowbit(i))
            ret += tr[i];
        return ret;
    }
    
    int minMovesToMakePalindrome(string s) {
        int n = s.size(), res = 0;
        
        // 预处理每个字符的位置
        vector<deque<int>> p(26);
        for (int i = 0; i < n; ++ i ) {
            p[s[i] - 'a'].push_back(i);
            add(i + 1, 1);  // 方便统计某个区间有多少个数
        }
        
        // t 已删除的个数
        int t = 0, odd = -1;
        for (int i = 0; i < n; ++ i ) {
            int c = s[i] - 'a';
            if (p[c].empty())
                continue;
            
            if (p[c].size() == 1) {
                // 奇数个
                odd = i;
                p[c].pop_back();
                continue;
            }
            
            // 总数 - 已删除的数 - 减去前面挖空的位置 ==> 后面数字的个数
            res += n - t - query(p[c].back() + 1);
            
            add(p[c].back() + 1, -1);   // 挖空当前
            p[c].pop_back(); p[c].pop_front();
            t ++ ;
        }
        if (odd != -1)
            res += n / 2 - query(odd);
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

> [!NOTE] **[Codeforces C. Palindrome Transformation](https://codeforces.com/problemset/problem/486/C)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 代码简单 推导较多 好题
> 
> - **回文串的几个重要特性**
> 
>  1. 要使成为回文串，即对应字符相等，无论变左边的 还是变右边的 亦或是左边变一点右边变一点，总步数都是一样的，所以差别就在于移动上
> 
>  2. 显然，全部动左边的或者全部动右边的，可以最优
> 
>  3. 为方便，统一动左边的，如果初始位置在右边，我们可以将它移到左边的对应位置（不计步数）
> 
>  4. 接下来算每一个对应的位置的最小步数
> 
>  5. 加上移动的最小步数就是最后的结果

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: C. Palindrome Transformation
// Contest: Codeforces - Codeforces Round #277 (Div. 2)
// URL: https://codeforces.com/problemset/problem/486/C
// Memory Limit: 256 MB
// Time Limit: 1000 ms
//
// Powered by CP Editor (https://cpeditor.org)

#include <bits/stdc++.h>
using namespace std;

// 思维题
// 一开始困于先后的移动顺序
//
// 左半部分和右半部分最后是对称的，因此我们发现一定只需要修改某一半的部分
// 这取决于 p 最开始在左半部分还是右半部分，显然当 p 最开始在左半部分时
// 我们只修改左半部分的字符，反之亦然。
int n, p;
string s;

int main() {
    cin >> n >> p >> s;
    int tn = n / 2;

    // 先累积需要修改的操作数 同时记录位置
    int res = 0;
    vector<int> ve;
    for (int i = 0; i < tn; ++i) {
        int t = abs(s[i] - s[n - i - 1]);
        t = min(t, 26 - t);

        res += t;
        if (t)
            // 哪个离起点近用哪个 【因为修改前后半段中对应位置的任意一个都可以】
            ve.push_back(abs(i + 1 - p) < abs(n - i - p) ? i + 1 : n - i);
    }
    sort(ve.begin(), ve.end());

    // 移动所需要的操作数
    int sz = ve.size();
    if (sz) {
        // 最远的 最近的 组成的中间段 是必然要挨个遍历的区间
        // 离起点最远的 减去最近的 为该必然要遍历的区间长度
        // 加上 到该区间的起点的最近距离
        int t = ve[sz - 1] - ve[0] + min(abs(ve[sz - 1] - p), abs(ve[0] - p));
        res += t;
    }

    cout << res << endl;

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

> [!NOTE] **[Codeforces B. Painting Eggs](https://codeforces.com/problemset/problem/282/B)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 500 的阈值设置很精妙 和 1000 相应导致本题做起来更简单

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: B. Painting Eggs
// Contest: Codeforces - Codeforces Round #173 (Div. 2)
// URL: https://codeforces.com/problemset/problem/282/B
// Memory Limit: 256 MB
// Time Limit: 5000 ms
//
// Powered by CP Editor (https://cpeditor.org)

#include <bits/stdc++.h>
using namespace std;

// 贪心 猜想错误
// https://codeforces.com/contest/282/submission/109770563
// https://codeforces.com/contest/282/submission/109770893
// 考虑投给 a 而不是 g 实际上对二者的总投资额差距都是造成了1000的影响
// 所以顺序和各个值就不那么重要了
//
// 按照题解思路：
// Start from the 1st egg and each time give the egg to A
//  if and only if giving it to A doesn't make the
//  difference > 500, otherwise give it to G.

int main() {
    // needed, otherwise TLE
    // https://codeforces.com/contest/282/submission/109775112
    std::ios::sync_with_stdio(false);
    std::cin.tie(0);

    int n;
    cin >> n;

    int tot = 0;
    for (int i = 0; i < n; ++i) {
        int a, b;
        cin >> a >> b;
        if (tot + a <= 500)
            tot += a, cout << 'A';
        else
            tot -= b, cout << 'G';
    }
    cout << endl;

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

> [!NOTE] **[Codeforces B. Playing Cubes](https://codeforces.com/problemset/problem/257/B)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 思维题 贪心

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: B. Playing Cubes
// Contest: Codeforces - Codeforces Round #159 (Div. 2)
// URL: https://codeforces.com/problemset/problem/257/B
// Memory Limit: 256 MB
// Time Limit: 2000 ms

#include <bits/stdc++.h>
using namespace std;

// 思维题
// ATTENTION 该角色放置的时候才得分
// 显然
//    对于第一个人：每次拿和之前一个相同的方块最优
//    对于第二个人：每次拿和之前一个不同的方块最优
// 第二个人最高时 min(n, m)：
//     ABABABAB-AAA...
//     BABABABA-AAA...
// 第一个人最高时 max(n, m)
//     AAAAABBBBB...B

int main() {
    int n, m;
    cin >> n >> m;

    int t1 = max(n, m) - 1, t2 = min(n, m);
    cout << t1 << ' ' << t2 << endl;

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

> [!NOTE] **[Codeforces A. Counting Kangaroos is Fun](https://codeforces.com/problemset/problem/372/A)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 思维题
> 
> 重点在于分析知最优的装载情况
> 
> 随后从中间开始贪心
> 
> 中间选取的位置 `n / 2 - 1` 很关键

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: A. Counting Kangaroos is Fun
// Contest: Codeforces - Codeforces Round #219 (Div. 1)
// URL: https://codeforces.com/problemset/problem/372/A
// Memory Limit: 256 MB
// Time Limit: 1000 ms

#include <bits/stdc++.h>
using namespace std;

// 直接从最后开始 或从最前面开始二分找都是错误思路
// https://codeforces.com/contest/372/submission/109908153
// 错误和这位博主相同：
//     http://www.voidcn.com/article/p-csyzprtj-bxn.html
//        另外一种错误的思路:
//        从大到小给每只袋鼠a[i]分配一个最大的且它能装得下的袋鼠a[j].
//        这种思路错误在于a[j]可能还可以给更小的袋鼠a[k]分配,
//        但是你把a[j]装下去了,可能除了a[j],a[i]之外没有其他袋鼠能装得下a[k]了。
//
// 正解：从中间开始贪心，【最小的k个 一定被最大的k个所包含】

const int N = 500010;

int n;
int a[N];

int main() {
    cin >> n;
    for (int i = 0; i < n; ++i)
        cin >> a[i];
    sort(a, a + n);

    int p = n - 1;
    // n / 2 - 1     ==>  初始化为止很重要
    // NOT n / 2
    // https://codeforces.com/contest/372/submission/109908762
    // AND NOT (n-1)/2
    // https://codeforces.com/contest/372/submission/109909096
    for (int i = n / 2 - 1; i >= 0; --i)
        if (a[i] * 2 <= a[p])
            --p;

    cout << p + 1 << endl;

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

> [!NOTE] **[Codeforces C. Median](https://codeforces.com/problemset/problem/166/C)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 思维题
> 
> 想到了每次增加会影响总个数 n 
> 
> 以及想到二分找区间
> 
> **没想到找到区间后直接在区间内添加这个数**
> 
> 使得计算得到的新的中点值仍是这个数
> 
> **写法实现很重要 重复做**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: C. Median
// Contest: Codeforces - Codeforces Round #113 (Div. 2)
// URL: https://codeforces.com/problemset/problem/166/C
// Memory Limit: 256 MB
// Time Limit: 2000 ms

#include <bits/stdc++.h>
using namespace std;

// (n + 1) / 2
// n 偶数 位置在前半部分最后一个
// n 奇数 位置在中间
//
// 0 ~ n-1 的话 取 (n - 1) / 2
const int N = 510;

int n, x;
int a[N];

int findL(int x) {
    int l = 0, r = n;
    while (l < r) {
        int m = l + r >> 1;
        if (a[m] < x)
            l = m + 1;
        else
            r = m;
    }
    return l;
}

int findR(int x) {
    int l = 0, r = n;
    while (l < r) {
        int m = l + r >> 1;
        if (a[m] <= x)
            l = m + 1;
        else
            r = m;
    }
    return l - 1;
}

int main() {
    cin >> n >> x;

    for (int i = 0; i < n; ++i)
        cin >> a[i];

    sort(a, a + n);

    int l = lower_bound(a, a + n, x) - a;
    int r = upper_bound(a, a + n, x) - a - 1;
    // int l = findL(x), r = findR(x);
    int mid = (n - 1) / 2;

    int res = 0;
    while (mid < l || mid > r) {
        ++res;
        ++r;
        mid = (++n - 1) / 2;
    }
    cout << res << endl;

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

> [!NOTE] **[Codeforces A. Fox and Box Accumulation](http://codeforces.com/problemset/problem/388/A)**
> 
> 题意: 
> 
> n 个箱子，每个箱子规定只能在它上面最多放 `a[i]` 的箱子，问最少放几列

> [!TIP] **思路**
> 
> 显然贪心排序
> 
> 线性做法需要思维 重复

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: A. Fox and Box Accumulation
// Contest: Codeforces - Codeforces Round #228 (Div. 1)
// URL: http://codeforces.com/problemset/problem/388/A
// Memory Limit: 256 MB
// Time Limit: 1000 ms

#include <bits/stdc++.h>
using namespace std;

const int N = 110;

int n;
int x[N];

int cnt = 0;
vector<int> p[N];

int main() {
    cin >> n;
    for (int i = 0; i < n; ++i)
        cin >> x[i];
    sort(x, x + n);

    int k = 0;
    for (int i = 0; i < n; ++i)
        if (k * (x[i] + 1) <= i)
            ++k;
    cout << k << endl;

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

> [!NOTE] **[Codeforces Devu and his Brother](http://codeforces.com/problemset/problem/439/D)** [TAG]
> 
> 题意: 
> 
> 给予两个数组 $a$ 和 $b$ ，每次操作可以使其中的任意元素加 $1$ 或减 $1$ 
> 
> 求使得 $a$ 中的最小值大于等于 $b$ 中的最大值的最小操作次数。

> [!TIP] **思路**
> 
> 起初猜测最终结果必然可以是 $a$ 或 $b$ 中出现过的数字，直接双指针维护（前缀和预处理）
> 
> 然后 WA...
> 
> 实际上，可以分别对 $a$ 升序 $b$ 降序排序，然后一一对比累计差值即可
> 
> **画图辅助理解**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: D. Devu and his Brother
// Contest: Codeforces - Codeforces Round #251 (Div. 2)
// URL: https://codeforces.com/problemset/problem/439/D
// Memory Limit: 256 MB
// Time Limit: 1000 ms

#include <bits/stdc++.h>
using namespace std;

using LL = long long;
const static int N = 1e5 + 10;

int n, m;
LL a[N], b[N];

int main() {
    cin >> n >> m;
    for (int i = 0; i < n; ++i)
        cin >> a[i];
    for (int i = 0; i < m; ++i)
        cin >> b[i];
    sort(a, a + n), sort(b, b + m, greater<int>());

    LL res = 0;
    for (int i = 0; i < n && i < m; ++i)
        if (a[i] >= b[i])
            break;
        else
            res += b[i] - a[i];

    cout << res << endl;

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

> [!NOTE] **[LeetCode 2234. 花园的最大总美丽值](https://leetcode-cn.com/problems/maximum-total-beauty-of-the-gardens/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 较显然的需要先枚举
> 
> **重点在于枚举哪一个维度**，加个双指针优化即可

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // flowers[i] <= 1e5, 可以枚举
    using LL = long long;
    
    long long maximumBeauty(vector<int>& flowers, long long newFlowers, int target, int full, int partial) {
        LL base = 0;
        vector<LL> fs;
        for (auto x : flowers)
            if (x >= target)
                base += full;
            else
                fs.push_back(x);
        fs.push_back(0);   // 0-th is 0
        sort(fs.begin(), fs.end());
        
        int n = fs.size() - 1;
        vector<LL> s(n + 1);
        for (int i = 1; i <= n; ++ i )
            s[i] = s[i - 1] + fs[i];
        
        // ATTENTION
        // 此时有两种枚举思路：
        // 1. 枚举【某个数值】以下的全部抹相同，剩下的尽可能选一些达到target的
        //      问题：因为full与partial大小关系，处理起来极为复杂
        // 2. 枚举【某个个数】的达到target，【其他的尽可能大，但都不超过target】==> 正确思路
        LL res = base;
        for (int i = 0, j = 0; i <= n; ++ i ) {
            LL cost = (LL)target * (n - i) - (s[n] - s[i]);
            if (cost > newFlowers)
                continue;
            LL left = newFlowers - cost;
            // 找到其他尽可能大的值
            while (j <= i && left >= (LL)fs[j] * j - s[j])
                j ++ ;
            
            LL c = 0;
            {
                // cout << " i = " << i << " cost = " << cost << " left = " << left << endl;
                int t = j - 1;
                if (t <= 0) // 需要做除数，不能为0
                    c = 0;
                else
                    c = min(LL(target - 1), (left + s[t]) / t);
            }
            // cout << " full number = " << n - i << " partial number = " << c << " val = " << (n - i) * full + c * partial << endl;
            
            res = max(res, base + (LL)(n - i) * full + c * partial);
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