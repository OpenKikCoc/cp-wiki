## 习题

> [!NOTE] **[AcWing 1416. 包装矩形](https://www.acwing.com/problem/content/1418/)**
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

const int N = 4;

PII rt[N];
int p[N] = {0, 1, 2, 3};
vector<PII> ans;

// 宽 高
void update(int a, int b) {
    if (a > b) swap(a, b);
    if (ans.empty() || a * b < ans[0].x * ans[0].y) ans = {{a, b}};
    else if (a * b == ans[0].x * ans[0].y) ans.push_back({a, b});
}

void work() {
    auto a = rt[p[0]], b = rt[p[1]], c = rt[p[2]], d = rt[p[3]];
    update(a.x + b.x + c.x + d.x, max(a.y, max(b.y, max(c.y, d.y))));
    update(max(a.x + b.x + c.x, d.x), d.y + max(a.y, max(b.y, c.y)));
    update(max(a.x + b.x, d.x) + c.x, max(max(a.y, b.y) + d.y, c.y));
    update(a.x + d.x + max(b.x, c.x), max(a.y, max(d.y, b.y + c.y)));
    update(max(a.x, d.x) + b.x + c.x, max(b.y, max(c.y, a.y + d.y)));
    if (b.x >= a.x && c.y >= b.y) {
        // 最后一种情况 最右侧底部的方块u较矮
        if (c.y < a.y + b.y) {
            if (d.x + a.x <= b.x + c.x)
                update(b.x + c.x, max(a.y + b.y, c.y + d.y));
        } else update(max(d.x, b.x + c.x), c.y + d.y);
    }
}

int main() {
    for (int i = 0; i < 4; ++ i ) cin >> rt[i].x >> rt[i].y;
    
    // 4 * 3 * 2 * 1 = 24 全排列
    for (int i = 0; i < 24; ++ i ) {
        // 每个矩形是否翻转
        for (int j = 0; j < 16; ++ j ) {
            for (int k = 0; k < 4; ++ k )
                if (j >> k & 1)
                    swap(rt[p[k]].x, rt[p[k]].y);
            work();
            for (int k = 0; k < 4; ++ k )
                if (j >> k & 1)
                    swap(rt[p[k]].x, rt[p[k]].y);
        }
        next_permutation(p, p + 4);
    }
    
    sort(ans.begin(), ans.end());
    ans.erase(unique(ans.begin(), ans.end()), ans.end());
    
    cout << ans[0].x * ans[0].y << endl;
    for (auto & a : ans) cout << a.x << ' ' << a.y << endl;
    
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

> [!NOTE] **[LeetCode 335. 路径交叉](https://leetcode-cn.com/problems/self-crossing/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 复杂分类讨论
>
> 我们分类讨论相交的情况有哪些。一共有三种：
>
> 连续的四条边相交：[![QQ图片20180628203655.png](https://camo.githubusercontent.com/5405978e0e0ab8b6bd68a103493561abb0b50a4f105def1458c05c0ba2690ee2/68747470733a2f2f7777772e616377696e672e636f6d2f6d656469612f61727469636c652f696d6167652f323031382f30362f32382f315f303933343038313237612d515125453525394225424525453725383925383732303138303632383230333635352e706e67)](https://camo.githubusercontent.com/5405978e0e0ab8b6bd68a103493561abb0b50a4f105def1458c05c0ba2690ee2/68747470733a2f2f7777772e616377696e672e636f6d2f6d656469612f61727469636c652f696d6167652f323031382f30362f32382f315f303933343038313237612d515125453525394225424525453725383925383732303138303632383230333635352e706e67)
>
> 连续的五条边相交：[![QQ图片20180628203840.png](https://camo.githubusercontent.com/50cdd44527f26c43314bd62d6b80be6e9389498a418926e1d4cf1de1910f8d87/68747470733a2f2f7777772e616377696e672e636f6d2f6d656469612f61727469636c652f696d6167652f323031382f30362f32382f315f333566326431663837612d515125453525394225424525453725383925383732303138303632383230333834302e706e67)](https://camo.githubusercontent.com/50cdd44527f26c43314bd62d6b80be6e9389498a418926e1d4cf1de1910f8d87/68747470733a2f2f7777772e616377696e672e636f6d2f6d656469612f61727469636c652f696d6167652f323031382f30362f32382f315f333566326431663837612d515125453525394225424525453725383925383732303138303632383230333834302e706e67)
>
> 连续的六条边相交：[![QQ图片20180628203957.png](https://camo.githubusercontent.com/9f7c299ccc9d45556b61f1559726b26793b07db7fec004e889ede44ba95cfc66/68747470733a2f2f7777772e616377696e672e636f6d2f6d656469612f61727469636c652f696d6167652f323031382f30362f32382f315f363433353638666137612d515125453525394225424525453725383925383732303138303632383230333935372e706e67)](https://camo.githubusercontent.com/9f7c299ccc9d45556b61f1559726b26793b07db7fec004e889ede44ba95cfc66/68747470733a2f2f7777772e616377696e672e636f6d2f6d656469612f61727469636c652f696d6167652f323031382f30362f32382f315f363433353638666137612d515125453525394225424525453725383925383732303138303632383230333935372e706e67)
>
> 然后遍历整个数组，判断这三种情况即可。

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    bool isSelfCrossing(vector<int>& x) {
        int n = x.size();
        if (n <= 3) return false;
        for (int i = 3; i < n; i ++ ) {
            if (x[i - 1] <= x[i - 3] && x[i] >= x[i - 2]) return true;
            if (i >= 4 && x[i - 3] == x[i - 1] && x[i] + x[i - 4] >= x[i - 2]) return true;
            if (i >= 5 && x[i - 3] >= x[i - 1] && x[i - 1] + x[i - 5] >= x[i - 3] && x[i - 2] >= x[i - 4] && x[i - 4] + x[i] >= x[i - 2])
                return true;
        }
        return false;
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

> [!NOTE] **[LeetCode 391. 完美矩形](https://leetcode-cn.com/problems/perfect-rectangle/)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> 所有点次数：
>
> |        |      |
> | ------ | ---- |
> | 1次    | 4    |
> | 3      | 0    |
> | 2 or 4 | inf  |
>
> 1次 4个
>
> 3   0
>
> 2or4 inf
>
> 总面积相同
>
> ==> 如果是完美矩形 那么一定满足两点：
>
> （1）最左下 最左上 最右下 最右上 的四个点只出现一次 其他点成对出现 
>
> （2）四个点围城的矩形面积 = 小矩形的面积之和
>
> ==> 把每个子矩形的面积累加，四个坐标放进一个vector，然后sort一下，
>
> 相同的坐标消去。最后剩下4个出现奇数次的点，且这个四个点围成的矩形面积等于子矩形面积和，则为true


<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    bool isRectangleCover(vector<vector<int>>& rectangles) {
        map<pair<int, int>, int> cnt;
        typedef long long LL;
        LL sum = 0;
        for (auto x : rectangles) {
            LL x1 = x[0], y1 = x[1], x2 = x[2], y2 = x[3];
            ++ cnt[{x1, y1}], ++ cnt[{x1, y2}];
            ++ cnt[{x2, y1}], ++ cnt[{x2, y2}];
            sum += (x2 - x1) * (y2 - y1);
        }
        vector<vector<int>> res;
        for (auto & [k, v] : cnt)
            if (v == 1) res.push_back({k.first, k.second});
            else if (v == 3) return false;
            else if (v > 4) return false;
        if (res.size() != 4) return false;
        sort(res.begin(), res.end());
        return sum == (LL)(res[3][0] - res[0][0]) * (res[3][1] - res[0][1]);
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

> [!NOTE] **[LeetCode 420. 强密码检验器](https://leetcode-cn.com/problems/strong-password-checker/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 复杂模拟...没啥意思
> 
> 分情况讨论后的实现

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int strongPasswordChecker(string s) {
        int a = 0, b = 0, c = 0, n = s.size(), k = 0;
        for (auto x: s) {
            if (x >= '0' && x <= '9') a = 1;
            else if (x >= 'a' && x <= 'z') b = 1;
            else if (x >= 'A' && x <= 'Z') c = 1;
        }
        k = a + b + c;
        if (n < 6) return max(6 - n, 3 - k);
        else {
            int p = 0;
            int d[3] = {0};
            // 1. 不能连续出现三次 否则必然需要每3个改一次
            //  推导知修改的步数最小 也即修改是最优操作
            //  插入: (k-1)/2 删除: k-2 修改: k/3
            for (int i = 0; i < s.size(); i ++ ) {
                int j = i;
                while (j < s.size() && s[j] == s[i]) j ++ ;
                // 1.1 取值
                int t = j - i;
                p += t / 3;
                // 1.2 所有长度大于等与3的连续段
                // 因为显然有 t/3 是向下取整
                // 故优先把余0的干掉,其次余1的,再次余2的
                if (t >= 3) d[t % 3] ++ ;
                // 更新 i
                i = j - 1;
            }
            if (n <= 20) return max(p, 3 - k);

            // 2. 还需要删
            //  则此时希望尽可能的用删除来覆盖1.1中的修改操作
            int del = n - 20, res = del;
            if (d[0] && del > 0) {
                // 删1个，同时使p减少相同数量
                int t = min(d[0], del);
                del -= t;
                p -= t;
            }
            if (d[1] && del > 0) {
                // 删2个，同时使p减少一半数量
                int t = min(d[1] * 2, del);
                del -= t;
                p -= t / 2;
            }
            if (p && del > 0) {
                // 特殊
                // 删3个
                int t = min(p * 3, del);
                p -= t / 3;
            }
            return res + max(p, 3 - k);
        }
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

> [!NOTE] **[LeetCode 2029. 石子游戏 IX](https://leetcode-cn.com/problems/stone-game-ix/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 一开始在想 区间DP
> 
> 其实是情况较复杂的分情况讨论
> 
> > 这题的题意设计很有意思 1 和 2 先后选取恰好对应 mod 3 的各类情况

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    bool stoneGameIX(vector<int>& stones) {
        int s[3] = {0, 0, 0};
        for (int i : stones)
            s[i % 3] ++ ;
        
        // s[0] 仅用作换手
        
        // 当 s[0] 为偶数，显然消除换手，只考虑 s[1] s[2] 即可
        // 如果 s[1] s[2] 任一为 0，则 alice 必败
        // ==> 分情况讨论
        //      s[1] = 0: alice 只能取 2 后面 bob 跟着取 2
        //                      后面 [取光] 或 [alice 三的倍数] 必败
        //      s[2] = 0: alice 只能取 1 后面 bob 跟着取 1
        //                      同理
        // 否则必胜
        if (s[0] % 2 == 0)
            return s[1] != 0 && s[2] != 0;
        
        // s[0] % 2 == 1 必然有一次换手
        // ==> 分情况讨论
        //      s[1] = s[2]: 则相当于 bob 先手选 s[1] s[2]
        //                   alice 为了跟上 bob 必须跟着取 最终取到最后石子(三的倍数) 必败
        //      abs(s[1] - s[2]) <= 2:  不管 alice 先取哪个 bob 都可以换手
        //                              最终石子取完 必败
        //      abs(s[1] - s[2]) > 2:   alice 取较多的 最终 bob 会到达三的倍数的情况 必胜
        return abs(s[1] - s[2]) > 2;
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

> [!NOTE] **[Codeforces C. Arithmetic Progression](http://codeforces.com/problemset/problem/382/C)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 巨多case和细节 简单就是麻烦

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: C. Arithmetic Progression
// Contest: Codeforces - Codeforces Round #224 (Div. 2)
// URL: http://codeforces.com/problemset/problem/382/C
// Memory Limit: 256 MB
// Time Limit: 1000 ms

#include <bits/stdc++.h>
using namespace std;

const int N = 100010;

int n;
int a[N];

using PII = pair<int, int>;

bool check(unordered_map<int, int>& S) {
    vector<PII> ve;
    for (auto& [k, v] : S)
        ve.push_back({k, v});
    sort(ve.begin(), ve.end());
    // 第三个条件
    // http://codeforces.com/contest/382/submission/110983876
    return ve[0].first == 0 || ve[1].second > 1 ||
           ve[1].first != ve[0].first * 2;
}

int main() {
    cin >> n;
    for (int i = 0; i < n; ++i)
        cin >> a[i];
    sort(a, a + n);

    if (n < 2)
        cout << -1 << endl;
    else if (n == 2) {
        int d = a[1] - a[0];
        if (d == 0) {
            cout << 1 << endl << a[0] << endl;
        } else if (d % 2 == 0) {
            cout << 3 << endl;
            cout << a[0] - d << ' ' << a[0] + d / 2 << ' ' << a[1] + d << endl;
        } else {
            cout << 2 << endl;
            cout << a[0] - d << ' ' << a[1] + d << endl;
        }
    } else {
        // n >= 3
        unordered_map<int, int> S;
        for (int i = 1; i < n; ++i)
            S[a[i] - a[i - 1]]++;
        if (S.size() > 2 || S.size() == 2 && check(S))
            cout << 0 << endl;
        else {
            int d = a[1] - a[0], d2 = d, p = 1;
            for (int i = 2; i < n; ++i) {
                d2 = a[i] - a[i - 1];
                p = i;
                if (d2 != d)
                    break;
            }

            if (d > d2) {
                cout << 1 << endl;
                cout << a[0] + d2 << endl;
            } else if (d < d2) {
                cout << 1 << endl;
                cout << a[p - 1] + d << endl;
            } else {
                // http://codeforces.com/contest/382/submission/110983324
                if (d) {
                    cout << 2 << endl;
                    cout << a[0] - d << ' ' << a[n - 1] + d << endl;
                } else {
                    cout << 1 << endl;
                    cout << a[0] << endl;
                }
            }
        }
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