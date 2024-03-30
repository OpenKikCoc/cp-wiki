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
> 
> 既是博弈论题，也是分情况讨论题

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

> [!NOTE] **[Codeforces Little Elephant and Interval](http://codeforces.com/problemset/problem/204/A)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 区间求值变为区间减
> 
> 分情况讨论 细节较多

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: A. Little Elephant and Interval
// Contest: Codeforces - Codeforces Round #129 (Div. 1)
// URL: https://codeforces.com/problemset/problem/204/A
// Memory Limit: 256 MB
// Time Limit: 2000 ms

#include <bits/stdc++.h>
using namespace std;

using LL = long long;
const static int N = 19;

LL l, r;

LL c[N], p[N];

void init() {
    p[0] = 1;
    for (int i = 1; i < N; ++i)
        p[i] = p[i - 1] * 10;

    c[1] = 10, c[2] = 9;
    for (int i = 3; i < N; ++i)
        c[i] = p[i - 2] * 9;
}

LL f(LL x) {
    if (x == 0)
        return 1;  // f[1] = 10
    string s = to_string(x);
    int n = s.size();

    LL ret = 0;
    for (int i = 1; i < n; ++i)
        ret += c[i];

    if (n > 2) {
        for (int i = 1; i < s[0] - '0'; ++i)
            ret += p[n - 2];
        LL t = stoll(s.substr(1, n - 2));
        if (s[0] > s[n - 1])
            ret += t;
        else
            ret += t + 1;
    } else if (n == 2) {
        if (s[0] > s[n - 1])
            ret += s[0] - '0' - 1;
        else
            ret += s[0] - '0';
    } else
        ret += s[0] - '0' + 1;
    return ret;
}

int main() {
    init();

    cin >> l >> r;

    cout << f(r) - f(l - 1) << endl;

    return 0;
}
```

##### **C++ 更简单**

```cpp
// Problem: A. Little Elephant and Interval
// Contest: Codeforces - Codeforces Round #129 (Div. 1)
// URL: https://codeforces.com/problemset/problem/204/A
// Memory Limit: 256 MB
// Time Limit: 2000 ms

#include <bits/stdc++.h>
using namespace std;

using LL = long long;

LL l, r;

LL f(LL x) {
    if (x < 10)
        return x;
    string s = to_string(x);
    // 可以想到对于一个大的范围，中间的数中，每十个数中会有一个满足要求。
    // 故中间直接计算，再加 9 表示首位的数值可能性
    LL base = (x / 10 + 9);
    // 首>尾 减一
    return base - (s[0] > s.back());
}

int main() {
    cin >> l >> r;

    cout << f(r) - f(l - 1) << endl;

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

> [!NOTE] **[LeetCode 2499. 让数组不相等的最小总代价](https://leetcode.cn/problems/minimum-total-cost-to-make-arrays-unequal/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> **非常好的分情况讨论题 需要严格的思维推导**
> 
> 假定: $x = nums1[i] = nums2[i]$
> 
> 1. x 的众数的次数 <= 需要调整的数字总数 / 2
> 
>    - 如果总数是偶数 内部两两匹配即可【思考 为什么直接两两匹配就行? 考虑每一个至少交换一次 而两两匹配恰好满足】
> 
>    - 如果总数是奇数 【ATTENTION】
>         
>      则出现的数字种数至少为 3 ==> 意味着必然有一种方式使得其中一个位置与 nums1[0] 交换剩下的两两匹配 【抽屉原理】
> 
> 2. x 的众数的次数 > 需要调整的数字总数 / 2
> 
>    两两匹配后多出的众数需要与借助其他（非需要调整的）位置交换
> 
>    在这些不需要调整的位置中 找到 nums1&nums2 都不为众数的 交换累加即可
> 
>    直到把所有多出的次数都交换完

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 分情况讨论
    // x = nums1[i] = nums2[i]
    // 1. x 的众数的次数 <= 需要调整的数字总数 / 2
    //     1.1 如果总数是偶数 内部两两匹配即可【思考 为什么直接两两匹配就行? 考虑每一个至少交换一次 而两两匹配恰好满足】
    //     1.2 如果总数是奇数 【ATTENTION】
    //         则出现的数字种数至少为 3 ==> 意味着必然有一种方式使得其中一个位置与 nums1[0] 交换剩下的两两匹配 【抽屉原理】
    // 2. x 的众数的次数 > 需要调整的数字总数 / 2
    //     两两匹配后多出的众数需要与借助其他（非需要调整的）位置交换
    //     在这些不需要调整的位置中 找到 nums1&nums2 都不为种数的 交换累加即可
    //     直到把所有多出的次数都交换完
    
    using LL = long long;
    const static int N = 1e5 + 10;
    
    int c[N];
    
    long long minimumTotalCost(vector<int>& nums1, vector<int>& nums2) {
        int n = nums1.size();
        
        memset(c, 0, sizeof c);
        int cnt = 0, mode = 0, mode_cnt = 0;
        LL res = 0;
        for (int i = 0; i < n; ++ i )
            if (nums1[i] == nums2[i]) {
                int x = nums1[i];
                cnt ++ ;            // 需要调整的位置 ++
                c[x] ++ ;           // 出现的数量 ++
                if (c[x] > mode_cnt)
                    // 更新众数及众数出现的数量
                    mode = x, mode_cnt = c[x];
                
                // 必不可少的代价
                res += i;
            }
        
        // 计算除了内部两两消化之外，还需借助其他位置的数量
        // 如果众数数量未过半，则 t <= 0 可以直接跳过后续计算流程
        int t = mode_cnt - (cnt - mode_cnt);
        for (int i = 0; i < n && t > 0; ++ i )
            // 合法的可借助的位置
            if (nums1[i] != nums2[i]) {
                if (nums1[i] != mode && nums2[i] != mode) {
                    res += i;
                    t -- ;
                }
            }
        
        // 如果还有未处理的
        if (t > 0)
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

> [!NOTE] **[LeetCode 2508. 添加边使所有节点度数都为偶数](https://leetcode.cn/problems/add-edges-to-make-degrees-of-all-nodes-even/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 经典分情况讨论
> 
> 需要理清楚 加快速度

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 思考:
    // +0 => 原图已经都是偶数
    // +1 => 原图恰好两个奇数, 且加完无重边
    // +2 => 4个度数分配 [1, 1, 1, 1] or [1, 1, 2]
    const static int N = 1e5 + 10;
    
    unordered_set<int> es[N];
    int d[N];
    
    bool has_edge(int a, int b) {
        return es[a].find(b) != es[a].end();
    }
    
    bool isPossible(int n, vector<vector<int>>& edges) {
        for (auto & e : edges)
            es[e[0]].insert(e[1]), es[e[1]].insert(e[0]), d[e[0]] ++ , d[e[1]] ++ ;
        
        vector<int> xs;
        for (int i = 1; i <= n; ++ i )
            if (d[i] & 1)
                xs.push_back(i);
        
        // +0
        {
            if (xs.empty())
                return true;
        }
        
        // +1
        {
            if (xs.size() == 2) {
                // 只要没有重边即可
                if (!has_edge(xs[0], xs[1]))
                    return true;
                // ATTENTION ==> 有重边则只要有个其他偶数边即可
                // 此时 has_edge 一定为 true，则需要让这两个点与其他点相连 且连接的点不应该出现在二者并集中
                for (int i = 1; i <= n; ++ i ) {
                    if (i == xs[0] || i == xs[1])
                        continue;
                    if (has_edge(xs[0], i) || has_edge(xs[1], i))
                        continue;
                    return true;
                }
            }
        }
        
        // +2
        {
            if (xs.size() == 4) {
                /*
                for (int i = 0; i < 4; ++ i )
                    for (int j = i + 1; j < 4; ++ j ) {
                        int a = xs[i], b = xs[j], c = -1, d = -1;
                        for (int k = 0; k < 4; ++ k )
                            if (k != i && k != j) {
                                if (c == -1)
                                    c = xs[k];
                                else
                                    d = xs[k];
                            }
                        
                        if (!has_edge(a, b) && !has_edge(c, d)) {
                            return true;
                        }
                    }
                */
                // 枚举过程可以优化
                int a = xs[0], b = xs[1], c = xs[2], d = xs[3];
                if (!has_edge(a, b) && !has_edge(c, d) ||
                    !has_edge(a, c) && !has_edge(b, d) ||
                    !has_edge(a, d) && !has_edge(b, c))
                    return true;
            }
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

> [!NOTE] **[LeetCode 2546. 执行逐位运算使字符串相等](https://leetcode.cn/problems/apply-bitwise-operations-to-make-strings-equal/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 结论：
> 
> - 1 可以把另一个 0 变成 1，或把另一个 1 变成 0
>
> - 如果存在 1，则无法彻底消除 1
>
> 进一步简化：**两个字符串要么都有 1，要么都没有 1**
> 
> 关键在于，如何得到简化结论？考虑什么情况下会失败：
>
> - 如果 s 存在 1，则如果 target 没有 1 必定失败
>
> - 如果 s 不存在 1，则如果 target 有 1 必定失败

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 0, 0 => 0, 0
    // 0, 1 => 1, 1
    // 1, 0 => 1, 1
    // 1, 1 => 1, 0 | 0, 1
    
    // 结论：1 可以把另一个 0 变成 1，或把另一个 1 变成 0
    //      但是无法彻底消除 1
    
    bool hasC(string & s, char c) {
        for (auto t : s)
            if (t == c)
                return true;
        return false;
    }
    
    bool makeStringsEqual(string s, string target) {
        bool f = hasC(s, '1');
        for (int i = 0; i < s.size(); ++ i )
            if (s[i] != target[i]) {
                // 如果有 1 则可以转换，不管是哪一个 1
                if (!f)
                    return false;
            }
        // 无法彻底消除 1
        if (f && !hasC(target, '1'))
            return false;
        return true;
    }
};
```

##### **C++ 简化**

```cpp
class Solution {
public:
    bool hasC(string & s, char c) {
        for (auto t : s)
            if (t == c)
                return true;
        return false;
    }

    bool makeStringsEqual(string s, string target) {
        return hasC(s, '1') == hasC(target, '1');
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

> [!NOTE] **[LeetCode 2749. 得到整数零需要执行的最少操作数](https://leetcode.cn/problems/minimum-operations-to-make-the-integer-zero/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 分情况讨论
> 
> 可以看下题解区其他解法 TODO

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using LL = long long;
    
    int makeTheIntegerZero(int num1, int num2) {
        LL x = num1, y = num2;
        for (int i = 1; i < 100; ++ i ) {
            LL t = x - y * i;
            if (t < 0)
                continue;       // ATTENTION 如果已经是负数肯定不能再减了
            
            // cout << " i = " << i << " t = " << t << endl;
            int x = i;
            if (t & 1)
                x -- , t -- ;
            
            int c = 0;
            for (int i = 0; i < 64; ++ i )  // ATTENTION 不止 32 位需要关注
                if (t >> i & 1)
                    c ++ ;
            
            // for (int i = 63; i >= 0; -- i )
            //     if (t >> i & 1)
            //         cout << '1';
            //     else
            //         cout << '0';
            // cout << endl;
            
            // ATTENTION 过滤条件
            if (c && x == 0)
                continue;
            if (c == 0 && x)
                continue;
            
            if (c <= x)
                return i;
        }
        return -1;
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

> [!NOTE] **[LeetCode 2982. 找出出现至少三次的最长特殊子字符串 II](https://leetcode.cn/problems/find-longest-special-substring-that-occurs-thrice-ii/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 分情况讨论 各个情况下的计算细节

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    const static int N = 30;
    
    vector<int> c[N];
    
    int maximumLength(string s) {
        int n = s.size();
        for (int i = 0; i < n; ++ i ) {
            int j = i;
            while (j < n && s[j] == s[i])
                j ++ ;
            c[s[i] - 'a'].push_back(j - i);
            i = j - 1;
        }
        
        int res = -1;
        for (int i = 0; i < N; ++ i ) {
            int m = c[i].size(), t = -1;
            if (m == 0)
                continue;
            
            sort(c[i].begin(), c[i].end());
            if (m >= 1)
                t = max(t, c[i][m - 1] - 2);
            if (m >= 2)
                t = max(t, min(c[i][m - 1] - 1, c[i][m - 2]));
            if (m >= 3)
                t = max(t, min({c[i][m - 3], c[i][m - 2], c[i][m - 1]}));
            
            // cout << " i = " << i << " ch = " << char('a' + i) << " t = " << t << " first " << c[i][0] << endl;
            // ATTENTION
            if (t)
                res = max(res, t);
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

> [!NOTE] **[LeetCode 2983. 回文串重新排列查询](https://leetcode.cn/problems/palindrome-rearrangement-queries/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 思维 暴力优化 分情况讨论

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
using PII = pair<int, int>;
const static int N = 1e5 + 10, M = 30;

// TLE 提到外面初始化
int sum1[N][M], sum2[N][M], diff[N];

int count_diff(vector<PII> & ps) {
    int ret = 0;
    for (auto & p : ps)
        ret += diff[p.second] - diff[p.first - 1];
    return ret;
}

class Solution {
public:
    int n;
    string s1, s2;

    void init() {
        for (int j = 0; j < 26; ++ j )
            sum1[0][j] = sum2[0][j] = 0;
        // memset(sum1, 0, sizeof sum1), memset(sum2, 0, sizeof sum2);
        for (int i = 1; i <= n; ++ i )
            for (int j = 0; j < 26; ++ j ) {
                sum1[i][j] = sum1[i - 1][j] + (j == s1[i - 1] - 'a');
                sum2[i][j] = sum2[i - 1][j] + (j == s2[i - 1] - 'a');
            }
        diff[0] = 0;
        for (int i = 1; i <= n; ++ i )
            diff[i] = diff[i - 1] + (s1[i - 1] != s2[i - 1]);
    }
    
    bool get(int l1, int r1, int l2, int r2) {
        vector<PII> cross, ind1, ind2;  // ind1: 区间 [l1,r1] 不相交的部分, ind2: 区间 [l2,r2] 不相交的部分
        if (max(l1, l2) <= min(r1, r2))
            cross.push_back({max(l1, l2), min(r1, r2)});
        if (cross.size() > 0) {
            {
                if (l1 < l2)
                    ind1.push_back({l1, l2 - 1});
                if (l1 > l2)
                    ind2.push_back({l2, l1 - 1});
            }
            {
                if (r1 > r2)
                    ind1.push_back({r2 + 1, r1});
                if (r1 < r2)
                    ind2.push_back({r1 + 1, r2});
            }
        } else {
            ind1.push_back({l1, r1});
            ind2.push_back({l2, r2});
        }
        
        // 1. 区间并集是否覆盖所有 `不同下标`
        if (count_diff(cross) + count_diff(ind1) + count_diff(ind2) < diff[n])
            return false;
        
        // 2. 计算每个子串有哪些可调配的字母
        static int c1[M], c2[M];
        for (int i = 0; i < 26; ++ i ) {
            c1[i] = sum1[r1][i] - sum1[l1 - 1][i];
            c2[i] = sum2[r2][i] - sum2[l2 - 1][i];
        }
        
        // 把第一/二个子串不得不调配的字母消耗掉
        for (auto & p : ind1)
            for (int i = 0; i < 26; ++ i )
                c1[i] -= sum2[p.second][i] - sum2[p.first - 1][i];
        for (auto & p : ind2)
            for (int i = 0; i < 26; ++ i )
                c2[i] -= sum1[p.second][i] - sum1[p.first - 1][i];
        
        // 如果出现字母不够 或剩余可调配字母不一样 则 false
        for (int i = 0; i < 26; ++ i )
            if (c1[i] < 0 || c2[i] < 0 || c1[i] != c2[i])
                return false;
        return true;
    }
    
    vector<bool> canMakePalindromeQueries(string s, vector<vector<int>>& queries) {
        this->n = s.size() / 2;
        {
            for (int i = 0; i < n; ++ i )
                s1.push_back(s[i]);
            for (int i = 0; i < n; ++ i )   // reverse
                s2.push_back(s[n * 2 - 1 - i]);
            init();
        }
        
        vector<bool> res;
        for (auto & q : queries)
            res.push_back(get(q[0] + 1, q[1] + 1, n * 2 - q[3], n * 2 - q[2]));
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

> [!NOTE] **[LeetCode 3017. 按距离统计房屋对数目 II](https://leetcode.cn/problems/count-the-number-of-houses-at-a-certain-distance-ii/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 非常考验思维的分情况讨论
> 
> 容易想到枚举思路，重点在于 case by case 的区间计算细节

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 考虑
    // - 如果是自环边 (或是相邻的二者的边) 相当于只有一个线性关系
    // - 如果是非自环边且非相邻二者 则 进一步考虑其造成的影响
    //      分三个部分，左[1, x], 中[x, y], 右边[y, n]
    //          各自内部均无影响
    //          左->中 左->右 均会由于新增边的存在 而距离缩短【相当于 <x 的部分整体向内部偏移】
    //                                                => 维护时 更适合用前缀和
    // 进一步 考虑枚举右端点 只计算左侧 最终结果*2即可

    using LL = long long;

    const static int N = 1e5 + 10;

    LL d[N];

    void update(int l, int r) {
        if (l > r)
            return;
        d[l] += 1, d[r + 1] -= 1;
    }

    vector<LL> get(int n) {
        vector<LL> res(n);
        for (int i = 1; i <= n; ++ i ) {
            d[i] += d[i - 1];
            res[i - 1] = d[i] * 2;
        }
        return res;
    }
    
    vector<long long> countOfPairs(int n, int x, int y) {
        memset(d, 0, sizeof d);
        if (x == y) {
            for (int i = 1; i <= n; ++ i )
                update(1, i - 1);
            return get(n);
        }
        if (x > y)
            swap(x, y);

        for (int i = 1; i <= n; ++ i ) {
            if (i <= x) {
                update(1, i - 1);
            } else if (i <= y) {
                if (2 * i <= x + y + 1)
                    update(1, i - 1);
                else {
                    // 最左端
                    update(2 + (y - i), x + (y - i)); // 不包含 x 位置本身  而是放到下面算

                    // 对于 [x, y] 内部的部分，不用想的太复杂
                    // 实际上这是个环，所以无论在哪个位置，总的距离和都是固定的
                    //      总共 y-x+1 个点
                    // ATTENTION 需要去除后续会重复的一小部分 (y-i)

                    int v = y - x;
                    update(1, v / 2), update(1 + (y - i) /*ATTENTION WA 因为i右边的不能计算 会重复*/, (v + 1) / 2);
                }
            } else {
                // 最左端
                update(2 + (i - y), x + (i - y));

                // [x,y) 中间的部分
                int v = y - x;
                update(1 + (i - y), v / 2 + (i - y)), update(1 + (i - y), (v + 1) / 2 + (i - y));

                // 右端 包含y
                update(1, i - y);
            }
        }
        
        return get(n);
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