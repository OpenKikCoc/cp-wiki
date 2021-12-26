区间类动态规划是线性动态规划的扩展，它在分阶段地划分问题时，与阶段中元素出现的顺序和由前一阶段的哪些元素合并而来有很大的关系。

令状态 $f(i,j)$ 表示将下标位置 $i$ 到 $j$ 的所有元素合并能获得的价值的最大值，那么 $f(i,j)=\max\{f(i,k)+f(k+1,j)+cost\}$，$cost$ 为将这两组元素合并起来的代价。

* * *

> [!NOTE] **区间 DP 的特点**
> 
> **合并**：即将两个或多个部分进行整合，当然也可以反过来；
> 
> **特征**：能将问题分解为能两两合并的形式；
> 
> **求解**：对整个问题设最优值，枚举合并点，将问题分解为左右两个部分，最后合并两个部分的最优值得到原问题的最优值。

* * *

## 习题



[NOIP 2007 矩阵取数游戏](https://vijos.org/p/1378)

[「IOI2000」邮局](https://www.luogu.com.cn/problem/P4767)


### 一维

> [!NOTE] **[AcWing 1068. 环形石子合并](https://www.acwing.com/problem/content/description/1070/)**
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
#include <algorithm>
#include <cstring>
#include <iostream>

using namespace std;

const int N = 410, INF = 0x3f3f3f3f;

int n;
int w[N], s[N];
int f[N][N], g[N][N];

int main() {
    cin >> n;
    for (int i = 1; i <= n; i++) {
        cin >> w[i];
        w[i + n] = w[i];
    }

    for (int i = 1; i <= n * 2; i++) s[i] = s[i - 1] + w[i];

    memset(f, 0x3f, sizeof f);
    memset(g, -0x3f, sizeof g);

    for (int len = 1; len <= n; len++)
        for (int l = 1; l + len - 1 <= n * 2; l++) {
            int r = l + len - 1;
            if (l == r)
                f[l][r] = g[l][r] = 0;
            else {
                for (int k = l; k < r; k++) {
                    f[l][r] =
                        min(f[l][r], f[l][k] + f[k + 1][r] + s[r] - s[l - 1]);
                    g[l][r] =
                        max(g[l][r], g[l][k] + g[k + 1][r] + s[r] - s[l - 1]);
                }
            }
        }

    int minv = INF, maxv = -INF;
    for (int i = 1; i <= n; i++) {
        minv = min(minv, f[i][i + n - 1]);
        maxv = max(maxv, g[i][i + n - 1]);
    }

    cout << minv << endl << maxv << endl;

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

> [!NOTE] **[AcWing 320. 能量项链](https://www.acwing.com/problem/content/description/322/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 变种 记忆整理

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <algorithm>
#include <cstring>
#include <iostream>

using namespace std;

const int N = 210, INF = 0x3f3f3f3f;

int n;
int w[N];
int f[N][N];

int main() {
    cin >> n;
    for (int i = 1; i <= n; i++) {
        cin >> w[i];
        w[i + n] = w[i];
    }
    // for(int i = 1; i <= n; ++i) f[i][i] = c[i];   // 错误
    // 定义左闭右开区间 len=3本质还是len=2 [1,n+1) = [1, n]
    for (int len = 3; len <= n + 1; len++)
        for (int l = 1; l + len - 1 <= n * 2; l++) {
            int r = l + len - 1;
            for (int k = l + 1; k < r; k++)
                f[l][r] = max(f[l][r], f[l][k] + f[k][r] + w[l] * w[k] * w[r]);
        }

    int res = 0;
    for (int l = 1; l <= n; l++) res = max(res, f[l][l + n]);

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

> [!NOTE] **[AcWing 479. 加分二叉树](https://www.acwing.com/problem/content/description/481/)**
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

const int N = 35;

int n;
int a[N];
int f[N][N], rt[N][N];

void dfs(int l, int r) {
    if (l > r) return;
    int v = rt[l][r];
    cout << v << ' ';
    dfs(l, v - 1);
    dfs(v + 1, r);
}

int main() {
    cin >> n;
    for (int i = 1; i <= n; ++ i )
        cin >> a[i];
    
    for (int i = 1; i <= n; ++ i )
        f[i][i] = a[i], rt[i][i] = i;
    
    for (int len = 2; len <= n; ++ len )
        for (int l = 1; l + len - 1 <= n; ++ l ) {
            int r = l + len - 1;
            for (int k = l; k <= r; ++ k ) {
                int left = k == l ? 1 : f[l][k - 1];
                int right = k == r ? 1 : f[k + 1][r];
                int score = left * right + a[k];
                if (f[l][r] < score) {
                    f[l][r] = score;
                    rt[l][r] = k;
                }
            }
        }
    
    cout << f[1][n] << endl;
    
    dfs(1, n);
    
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

> [!NOTE] **[AcWing 1069. 凸多边形的划分](https://www.acwing.com/problem/content/description/1071/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 1~n 就覆盖了所有情况 无需重复2n
> 
> 高精度

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <algorithm>
#include <cstring>
#include <iostream>

using namespace std;

typedef long long LL;

const int N = 55, M = 35, INF = 1e9;

int n;
int w[N];
LL f[N][N][M];

void add(LL a[], LL b[]) {
    static LL c[M];
    memset(c, 0, sizeof c);
    for (int i = 0, t = 0; i < M; i++) {
        t += a[i] + b[i];
        c[i] = t % 10;
        t /= 10;
    }
    memcpy(a, c, sizeof c);
}

void mul(LL a[], LL b) {
    static LL c[M];
    memset(c, 0, sizeof c);
    LL t = 0;
    for (int i = 0; i < M; i++) {
        t += a[i] * b;
        c[i] = t % 10;
        t /= 10;
    }
    memcpy(a, c, sizeof c);
}

int cmp(LL a[], LL b[]) {
    for (int i = M - 1; i >= 0; i--)
        if (a[i] > b[i])
            return 1;
        else if (a[i] < b[i])
            return -1;
    return 0;
}

void print(LL a[]) {
    int k = M - 1;
    while (k && !a[k]) k--;
    while (k >= 0) cout << a[k--];
    cout << endl;
}

int main() {
    cin >> n;
    for (int i = 1; i <= n; i++) cin >> w[i];

    LL temp[M];
    for (int len = 3; len <= n; len++)
        for (int l = 1; l + len - 1 <= n; l++) {
            int r = l + len - 1;
            f[l][r][M - 1] = 1;
            for (int k = l + 1; k < r; k++) {
                memset(temp, 0, sizeof temp);
                temp[0] = w[l];
                mul(temp, w[k]);
                mul(temp, w[r]);
                add(temp, f[l][k]);
                add(temp, f[k][r]);
                if (cmp(f[l][r], temp) > 0) memcpy(f[l][r], temp, sizeof temp);
            }
        }

    print(f[1][n]);

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

> [!NOTE] **[Luogu [HNOI2010]合唱队](https://www.luogu.com.cn/problem/P3205)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 统计方案数 流程

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

// 显然区间dp 如下实现即可
//
// 对于子区间长度为1的情况写了好几个if判断 不够优雅
// 实际上只有一个人的方案只有一种 可以直接按照默认在左侧进来
// 即 初始化 f[i][i][0] = 1, f[i][i][1] = 0;即可 略

const int N = 1010, MOD = 19650827;

int n;
int a[N];
int f[N][N][2];

int main() {
    cin >> n;
    for (int i = 1; i <= n; ++ i )
        cin >> a[i];
    
    for (int i = 1; i <= n; ++ i )
        f[i][i][0] = f[i][i][1] = 1;
    
    for (int len = 2; len <= n; ++ len )
        for (int l = 1; l + len - 1 <= n; ++ l ) {
            int r = l + len - 1;
            
            {
                int v = a[l];
                int tl = l + 1, tr = r, vl = 0, vr = 0;
                if (v < a[tl])
                    vl = f[tl][tr][0];
                if (v < a[tr])
                    vr = f[tl][tr][1];
                if (tl == tr)
                    f[l][r][0] = (f[l][r][0] + vl) % MOD;
                else
                    f[l][r][0] = (f[l][r][0] + (vl + vr) % MOD) % MOD;
            }
            {
                int v = a[r];
                int tl = l, tr = r - 1, vl = 0, vr = 0;
                if (v > a[tl])
                    vl = f[tl][tr][0];
                if (v > a[tr])
                    vr = f[tl][tr][1];
                if (tl == tr)
                    f[l][r][1] = (f[l][r][1] + vl) % MOD;
                else
                    f[l][r][1] = (f[l][r][1] + (vl + vr) % MOD) % MOD;
            }
        }
        
    cout << (f[1][n][0] + f[1][n][1]) % MOD << endl;
    
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

> [!NOTE] **[Luogu Zuma](https://www.luogu.com.cn/problem/CF607B)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 经典
> 
> 题面回文但非回文处理

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 510, INF = 0x3f3f3f3f;

int n, a[N], f[N][N];

int main() {
    cin >> n;
    for (int i = 1; i <= n; ++ i )
        cin >> a[i];
    
    memset(f, 0x3f, sizeof f);
    for (int i = 1; i <= n; ++ i )
        f[i][i] = 1;
    
    for (int len = 2; len <= n; ++ len )
        for (int l = 1; l + len - 1 <= n; ++ l ) {
            int r = l + len - 1;
            if (l + 1 == r) {
                // f[l][r - 1] = f[l + 1][r] = 1;
                f[l][r] = min(f[l][r - 1], f[l + 1][r]) + (a[l] != a[r]);
            } else {
                if (a[l] == a[r])
                    f[l][r] = f[l + 1][r - 1];
                for (int k = l; k < r; ++ k )
                    f[l][r] = min(f[l][r], f[l][k] + f[k + 1][r]);
            }
        }
    cout << f[1][n] << endl;
    
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

### 二维

> [!NOTE] **[AcWing 321. 棋盘分割](https://www.acwing.com/problem/content/description/323/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 二维分割问题
> 
> 注意状态定义 记忆化搜索

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

using namespace std;

const int N = 15, M = 9;
const double INF = 1e9;

int n, m = 8;
int s[M][M];
double f[M][M][M][M][N];
double X;

int get_sum(int x1, int y1, int x2, int y2) {
    return s[x2][y2] - s[x2][y1 - 1] - s[x1 - 1][y2] + s[x1 - 1][y1 - 1];
}

double get(int x1, int y1, int x2, int y2) {
    double sum = get_sum(x1, y1, x2, y2) - X;
    return (double)sum * sum / n;
}

double dp(int x1, int y1, int x2, int y2, int k) {
    double &v = f[x1][y1][x2][y2][k];
    if (v >= 0) return v;
    if (k == 1) return v = get(x1, y1, x2, y2);

    v = INF;
    for (int i = x1; i < x2; i++) {
        v = min(v, get(x1, y1, i, y2) + dp(i + 1, y1, x2, y2, k - 1));
        v = min(v, get(i + 1, y1, x2, y2) + dp(x1, y1, i, y2, k - 1));
    }

    for (int i = y1; i < y2; i++) {
        v = min(v, get(x1, y1, x2, i) + dp(x1, i + 1, x2, y2, k - 1));
        v = min(v, get(x1, i + 1, x2, y2) + dp(x1, y1, x2, i, k - 1));
    }

    return v;
}

int main() {
    cin >> n;
    for (int i = 1; i <= m; i++)
        for (int j = 1; j <= m; j++) {
            cin >> s[i][j];
            s[i][j] += s[i - 1][j] + s[i][j - 1] - s[i - 1][j - 1];
        }

    X = (double)s[m][m] / n;
    memset(f, -1, sizeof f);
    printf("%.3lf\n", sqrt(dp(1, 1, 8, 8, n)));

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

### 进阶

> [!NOTE] **[LeetCode 546 移除盒子](https://github.com/OpenKikCoc/LeetCode/blob/master/0501-0600/0546/README.md)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> [acwing 题解](https://www.acwing.com/solution/content/6168/)
> 
> 区间DP进阶。基础版的问题为：最少需要多少次操作（操作的规则一样）能删除掉所有的盒子。
> 
> TODO 重复做



<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    const static int N = 110;

    // f[i][j][k] 所有将区间 [i, j] 清空，且最后删除i，且最后删除时的长度为k的最大取值
    // g[i][j] = max(f[i][j][0], f[i][j][1] ... f[i][j][k])
    int f[N][N][N], g[N][N];

    int removeBoxes(vector<int>& boxes) {
        int n = boxes.size();
        memset(f, 0xcf, sizeof f);
        memset(g, 0xcf, sizeof g);

        for (int len = 1; len <= n; ++ len )
            for (int l = 0; l + len - 1 < n; ++ l ) {
                int r = l + len - 1;
                for (int k = 1; k <= len; ++ k ) {
                    if (len == 1)
                        f[l][r][k] = 1;
                    else if (k == 1)
                        f[l][r][k] = 1 + g[l + 1][r];
                    else
                        // 枚举时因为 l 总是第一个被删除的数
                        // 因此不同点设置为【第二个会被删除的数】并以其作为 u 
                        // 此时 总取值为
                        //   `l`    +     `range_l`       +  `u as the beginning of range_r`
                        //   `l`    +  `g[l + 1][u - 1]`  +  `f[u][r][k - 1]`
                        // 又因为最终删除时 l 与 range_r 一体
                        //      其价值为 f[u][r][k-1] - (k-1)*(k-1) + k*k 【思考】
                        // 综上有以下代码实现
                        for (int u = l + 1; u <= r - k + 2; ++ u ) {
                            if (boxes[u] != boxes[l])
                                continue;
                            int t = 0;
                            if (l + 1 <= u - 1)
                                t = g[l + 1][u - 1];
                            f[l][r][k] = max(f[l][r][k], t + f[u][r][k - 1] - (k - 1) * (k - 1) + k * k);
                        }
                    g[l][r] = max(g[l][r], f[l][r][k]);
                }
            }
        return g[0][n - 1];
    }
};
```

##### **C++ 新理解**

```cpp
class Solution {
public:
    const static int N = 110;

    int f[N][N][N], g[N][N];

    int removeBoxes(vector<int>& boxes) {
        int n = boxes.size();
        memset(f, 0xcf, sizeof f);
        memset(g, 0xcf, sizeof g);

        for (int i = 0; i < n; ++ i )
            g[i][i] = f[i][i][1] = 1;

        for (int len = 2; len <= n; ++ len )
            for (int l = 0; l + len - 1 < n; ++ l ) {
                int r = l + len - 1;
                for (int k = 1; k <= len; ++ k ) {
                    if (k == 1)
                        f[l][r][k] = 1 + g[l + 1][r];
                    else
                        // u 可以取到最右侧时为 u + （k - 1) - 1 == r
                        // 也即 u == r - (k - 1) + 1
                        for (int u = l + 1; u <= r - (k - 1) + 1; ++ u ) {
                            if (boxes[u] != boxes[l])
                                continue;
                            int t = 0;
                            if (l + 1 <= u - 1)
                                t = g[l + 1][u - 1];
                            f[l][r][k] = max(f[l][r][k], t + f[u][r][k - 1] - (k - 1) * (k - 1) + k * k);
                        }
                    g[l][r] = max(g[l][r], f[l][r][k]);
                }
            }
        return g[0][n - 1];
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

> [!NOTE] **[LeetCode 2019. 解出数学表达式的学生分数](https://leetcode-cn.com/problems/the-score-of-students-solving-math-expression/)**
> 
> [weekly-260](https://github.com/OpenKikCoc/LeetCode/tree/master/Contest/2021-09-26_Weekly-260)
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 区间DP进阶。中缀表达式计算 + 区间DP + 剪枝

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    stack<int> num;
    stack<char> op;
    void eval() {
        auto a = num.top(); num.pop();
        auto b = num.top(); num.pop();
        auto c = op.top(); op.pop();
        int r;
        if (c == '+')
            r = a + b;
        else
            r = a * b;
        num.push(r);
    }
    int calc(string s) {
        unordered_map<char, int> pr;
        pr['+'] = pr['-'] = 1, pr['*'] = pr['/'] = 2;
        for (int i = 0; i < s.size(); ++ i ) {
            char c = s[i];
            if (c == ' ')
                continue;
            if (isdigit(c)) {
                int x = 0, j = i;
                while (j < s.size() && isdigit(s[j]))
                    x = x * 10 + s[j] - '0', j ++ ;
                num.push(x);
                i = j - 1;
            } else {
                while (op.size() && pr[op.top()] >= pr[c])
                    eval();
                op.push(c);
            }
        }
        while (op.size())
            eval();
        return num.top();
    }

    int scoreOfStudents(string s, vector<int>& answers) {
        int tar = calc(s), n = s.size();

        unordered_set<int> f[32][32];
        for (int len = 1; len <= n; ++ len )
            for (int l = 0; l + len - 1 < n; l += 2 ) {
                int r = l + len - 1;
                if (l == r)
                    f[l][r].insert(s[l] - '0');
                else {
                    for (int k = l; k < r; k += 2)
                        for (auto v1 : f[l][k])
                            for (auto v2 : f[k + 2][r]) {
                                int t = 0;
                                if (s[k + 1] == '+')
                                    t = v1 + v2;
                                else
                                    t = v1 * v2;
                                if (t > 1000)
                                    continue;
                                f[l][r].insert(t);
                            }
                }
            }
        
        int res = 0;
        for (auto v : answers)
            if (v == tar)
                res += 5;
            else if (f[0][n - 1].count(v))
                res += 2;
        return res;
    }
};
```


##### **C++ 初版TLE**

```cpp
// TLE
class Solution {
public:
    using PON = pair<vector<char>, vector<int>>;
    using PONI = pair<PON, int>;
        
    int n, tar;
    unordered_set<int> S;
    set<PONI> hash;
    
    PON parse(string s) {
        vector<char> ops;
        vector<int> nums;
        int n = s.size();
        for (int i = 0; i < n; ++ i ) {
            int j = i, v = 0;
            while (j < n && isdigit(s[j]))
                v = v * 10 + s[j] - '0', j ++ ;
            nums.push_back(v);
            if (j < n)
                ops.push_back(s[j]);
            i = j;
        }
        return {ops, nums};
    }
    
    int op(char c, int a, int b) {
        if (c == '+')
            return a + b;
        return a * b;
    }
    
    void dfs(vector<char> ops, vector<int> nums, int cnt) {
        // MEM
        PONI t = {{ops, nums}, cnt};
        if (hash.count(t))
            return;
        hash.insert(t);
        
        if (ops.empty()) {
            // cout << "cnt = " << cnt << " nums[0] = " << nums[0] << endl;
            if (cnt == 0)
                this->tar = nums[0];
            else
                S.insert(nums[0]);
            return;
        }
        
        int n = ops.size(), p = 0;
        for (int i = 0; i < n; ++ i )
            if (ops[i] == '*') {
                p = i;
                break;
            }
        
        for (int i = 0; i < n; ++ i ) {
            int v = op(ops[i], nums[i], nums[i + 1]);
            // ATTENTION 增加一个剪枝
            if (v >= 1000)
                continue;
            
            vector<char> t_ops;
            vector<int> t_nums;
            for (int j = 0; j < i; ++ j )
                t_ops.push_back(ops[j]), t_nums.push_back(nums[j]);
            t_nums.push_back(v);
            for (int j = i + 1; j < n; ++ j )
                t_ops.push_back(ops[j]), t_nums.push_back(nums[j + 1]);
            dfs(t_ops, t_nums, cnt + (i != p));
        }
    }
    
    int scoreOfStudents(string s, vector<int>& answers) {
        this->n = s.size();
        auto [ops, nums] = parse(s);
        
        dfs(ops, nums, 0);
        
        // cout << "tar = " << tar << endl;
        
        int res = 0;
        for (auto v : answers)
            if (v == tar)
                res += 5;
            else if (S.count(v))
                res += 2;
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

### 优化进阶

尤其是线性优化

> [!NOTE] **[LeetCode 1937. 扣分后的最大得分](https://leetcode-cn.com/problems/maximum-number-of-points-with-cost/)**
> 
> [weekly-250](https://github.com/OpenKikCoc/LeetCode/tree/master/Contest/2021-07-18_Weekly-250)
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 本行状态依赖于上一行的状态，且本行内状态时前后有关联关系（区间修改/移动）
> 
> 尝试根据状态转移方程【拆掉绝对值表达式】
> 
> $O(nm^2)$ 优化为 $O(nm)$

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using LL = long long;
    const int INF = 0x3f3f3f3f;
    int n, m;
    vector<vector<int>> ps;
    vector<LL> f, g;
    
    long long maxPoints(vector<vector<int>>& points) {
        this->ps = points;
        this->n = ps.size(), this->m = ps[0].size();
        
        f = g = vector<LL>(m);
        
        for (int i = 0; i < m; ++ i )
            f[i] = ps[0][i];
        
        for (int i = 1; i < n; ++ i ) {
            g = f;
            {
                LL maxv = -INF;
                for (int j = 0; j < m; ++ j ) {
                    maxv = max(maxv, g[j] + j);
                    f[j] = max(f[j], ps[i][j] - j + maxv);
                }
            }
            {
                LL maxv = -INF;
                for (int j = m - 1; j >= 0; -- j ) {
                    maxv = max(maxv, g[j] - j);
                    f[j] = max(f[j], ps[i][j] + j + maxv);
                }
            }
        }
        return *max_element(f.begin(), f.end());
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

> [!NOTE] **[LeetCode 1977. 划分数字的方案数](https://leetcode-cn.com/problems/number-of-ways-to-separate-numbers/)**
> 
> [biweekly-59](https://github.com/OpenKikCoc/LeetCode/tree/master/Contest/2021-08-21_Biweekly-59)
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 当前 $f[i][j]$ 计算依赖于 $f[k][1 ~ j-1]$ 的求和 ，考虑前缀和优化
> 
> 另外需要字符串比较，使用 LCP 的思路【第一次见 LCP】



<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using LL = long long;
    const int MOD = 1e9 + 7;
    
    string s;
    vector<vector<int>> f, sum, lcp;
    
    // [r1结束长度为l的串] 是否 >= [r2结束长度为l的串]
    bool check(int r1, int r2, int l) {
        int l1 = r1 - l + 1, l2 = r2 - l + 1;
        if (l1 <= 0 || l2 <= 0)
            return false;
        int t = lcp[l1][l2];
        return t >= l || s[l1 + t - 1] > s[l2 + t - 1];
    }
    
    int numberOfCombinations(string num) {
        this->s = num;
        int n = s.size();
        f = sum = lcp = vector<vector<int>>(n + 1, vector<int>(n + 1));
        
        // lcp
        for (int i = n; i; -- i )
            for (int j = n; j; -- j )
                if (s[i - 1] != s[j - 1])
                    lcp[i][j] = 0;
                else {
                    lcp[i][j] = 1;
                    if (i < n && j < n)
                        lcp[i][j] += lcp[i + 1][j + 1];
                }
        
        // 初始化
        f[0][0] = 1;
        for (int i = 0; i <= n; ++ i )
            sum[0][i] = 1;  // sum[0][i] = sum[0][i - 1]
        
        // f[i][j] 前i个数 最后一个长度为j的方案数
        // sum[i][j] 以i结尾 长度不超过j的方案数总和
        for (int i = 1; i <= n; ++ i ) {
            for (int j = 1; j <= i; ++ j ) {
                int k = i - j;
                // 前缀和优化 将[枚举k结尾长度]的On降为O1
                if (s[k + 1 - 1] == '0')
                    f[i][j] = 0;    // 本段包含前缀0 非法
                else {
                    // case 1 长度小于j的都合法
                    f[i][j] = sum[k][j - 1];
                    // for (int t = 0; t < j; ++ t )
                    //     f[i][j] += f[k][t];
                    
                    // case 2 长度等于j的要比较大小
                    if (check(i, k, j))
                        f[i][j] = (f[i][j] + f[k][j]) % MOD;
                }
                // 更新
                sum[i][j] = (sum[i][j - 1] + f[i][j]) % MOD;
            }
            // 更新 根据定义，且j在内层循环所以必须这么写
            for (int j = i + 1; j <= n; ++ j )
                sum[i][j] = sum[i][j - 1];
        }
        int res = 0;
        for (int i = 1; i <= n; ++ i )
            res = (res + f[n][i]) % MOD;    // add
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