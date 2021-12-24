## 习题

> [!NOTE] **[AcWing 1379. 联系](https://www.acwing.com/problem/content/1381/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 写法 **前面加一个1的技巧**
> 
> 很多类似的 trick 技巧

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 200010, M = 1 << 13;

int A, B, n, m;
int cnt[M];
struct Data {
    // 出现次数 数值
    int k, num;
    bool operator< (const Data & t) const {
        if (k != t.k) return k > t.k;
        return num < t.num;
    }
    string get_string() {
        string res;
        for (int i = num; i; i >>= 1 )
            res += i % 2 + '0';
        // 删掉最高位的1
        res.pop_back();
        reverse(res.begin(), res.end());
        return res;
    }
}q[N];

int main() {
    cin >> A >> B >> m;
    
    string str, line;
    while (cin >> line) str += line;
    
    int n = str.size();
    for (int i = A; i <= B; ++ i )
        for (int j = 0, x = 0; j < n; ++ j ) {
            x = x * 2 + str[j] - '0';
            if (j - i >= 0) x -= str[j - i] - '0' << i;
            if (j >= i - 1) cnt[x + (1 << i)] ++ ;  // + (1 << i) 补一个1 避免相同大小数值但长度不同的串无法区分
        }
    
    for (int i = 0; i < M; ++ i ) q[i] = {cnt[i], i};
    
    sort(q, q + M);
    for (int i = 0, k = 0; i < M && k < m; ++ i , ++ k ) {
        if (!q[i].k) break;
        int j = i;
        while (j < M && q[j].k == q[i].k) ++ j ;
        cout << q[i].k << endl;
        for (int a = i, b = 0; a < j; ++ a , ++ b ) {
            cout << q[a].get_string();
            if ((b + 1) % 6 == 0) cout << endl;
            else cout << ' ';
        }
        if ((j - i) % 6) cout << endl;
        i = j - 1;
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

> [!NOTE] **[AcWing 1396. 街头竞速](https://www.acwing.com/problem/content/1398/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> **思维题**
> 
> 起点、终点分别可以达到任意点
> 
> 求哪些是必经点:
> 
> - 枚举 删点后检查能否起点到终点
> 
> 求哪些是分割点:
> 
> - 某个点可以将图分为两个不存在公共边的两部分 且两部分都合规
> 
>   也即 该点既是前半部分的终点也是后半部分的起点
> 
>   直接在必经点中找

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 55;

int n;
bool g[N][N], st1[N], st2[N];

void dfs(int u, int p, bool st[]) {
    st[u] = true;
    for (int i = 0; i <= n; ++ i )
        if (!st[i] && g[u][i] && i != p)
            dfs(i, p, st);
}

int main() {
    int b;
    while (cin >> b, b != -1) {
        if (b != -2) {
            g[n][b] = true;
            while (cin >> b, b != -2) g[n][b] = true;
        }
        n ++ ;
    }
    // 终点是 n
    n -- ;
    
    vector<int> res1, res2;
    for (int i = 1; i < n; ++ i ) {
        memset(st1, 0, sizeof st1);
        memset(st2, 0, sizeof st2);
        dfs(0, i, st1);
        // 如果删掉 i 无法访问到终点 n 
        if (!st1[n]) {
            res1.push_back(i);
            dfs(i, -1, st2);
            bool flag = true;
            // 如果 st1[j] = st2[j] = true 说明可以遍历回左边 显然就不是分割点
            for (int j = 0; j < n; ++ j )
                if (j != i && st1[j] && st2[j]) {
                    flag = false;
                    break;
                }
            if (flag) res2.push_back(i);
        }
    }
    
    cout << res1.size();
    for (auto x : res1) cout << ' ' << x;
    cout << endl << res2.size();
    for (auto x : res2) cout << ' ' << x;
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

> [!NOTE] **[Luogu 单词覆盖还原](https://www.luogu.com.cn/problem/P1321)**
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

int boy, girl;
string x;

int main() {
   cin >> x;
   for (int i = 0; i < x.length(); i ++ ){
   	// boy
   	if (x[i] == 'b') boy ++ ;
   	if (x[i] == 'o' && x[i - 1] != 'b') boy ++ ;
   	if (x[i] == 'y' && x[i - 1] != 'o') boy ++ ;
   	// girl
   	if (x[i] == 'g') girl ++;
   	if (x[i] == 'i' && x[i - 1] != 'g') girl ++ ;
   	if (x[i] == 'r' && x[i - 1] != 'i') girl ++ ;
   	if (x[i] == 'l' && x[i - 1] != 'r') girl ++ ;
   }
   cout << boy << endl << girl << endl;
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

### 剪绳子

> [!NOTE] **[SwordOffer 14- I. 剪绳子](https://leetcode-cn.com/problems/jian-sheng-zi-lcof/)**
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
    int maxProductAfterCutting(int n) {
        if (n <= 3)
            return n - 1;
        
        int res = 1;
        if (n % 3 == 1)
            res *= 4, n -= 4;
        while (n >= 3)
            res *= 3, n -= 3;
        if (n == 2)
            res *= 2;
        return res;
    }
};
```

##### **Python**

```python
# python3
# dp
# 状态表示：f[i] : 表示长度为i时的乘积方案数；属性：Max
# 状态转移：第一刀剪在长度为j, 那区别在于：后面的（i-j)是否还要再剪：
#          要剪：那就是j * f[i-j];不剪：j *(i-j)
class Solution:
    def cuttingRope(self, n: int) -> int:
        # 长度为1时，为0；长度为2，最大乘积为1
        f = [0] * (n + 1)  
        for i in range(2, n + 1):
            for j in range(1, i):
                f[i] = max(f[i], j * (i-j), j * f[i-j])
        return f[n]
      
 

# 数学方法
# 结论：把这个整数分成尽可能多的3，如果剩下两个2，就分成两个2 
# 证明如下：
# 1. 显然1 不会出现在最优解里
# 2. 证明最优解没有 大于等于5的数。假设有一个数ni >= 5, 那么其实可以把ni 拆为 3 + （ni -3），很显然可以证明：3(ni-3) > ni；所以最优解里肯定不会有大于等于5的数字，那最大的只可能是4
# 3. 证明最优解里不存在4，因为 4 拆出来 2 + 2，乘积不变，所以可以定最优解里没有4
# 4. 证明最优解里最多只能有两个2，因为假设有3个2，那3 * 3 > 2 * 2 * 2, 替换成3 乘积更大，所以最优解不能有三个2.

# 综上，选用尽量多的3，直到剩下2 或者 4，用2.

class Solution:
    def cuttingRope(self, n: int) -> int:
        if n <= 3:return 1 * (n - 1)
        # 踩坑：res 初始化位1 
       	res = 1
        # 处理 最优解 有两个2的情况
        if n % 3 == 1:  
            res *= 4
            n -= 4
         # 处理 最优解 只有一个2的情况
        if n % 3 == 2:  
            res *= 2
            n -= 2
        while n:
            res *= 3
            n -= 3
        return res
```

<!-- tabs:end -->
</details>

<br>

* * *

### dfs

> [!NOTE] **[Luogu 幻象迷宫](https://www.luogu.com.cn/problem/P1363)**
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

const int N = 1510;

int n, m;
int vis[N][N][3];
bool flag, g[N][N];

int dx[4] = {1, -1, 0, 0}, dy[4] = {0, 0, 1, -1};

void dfs(int x, int y, int lx, int ly) {
    if (flag)
        return;
    if (vis[x][y][0] && (vis[x][y][1] != lx || vis[x][y][2] != ly)) {
        flag = 1;
        return;
    }
    vis[x][y][1] = lx, vis[x][y][2] = ly, vis[x][y][0] = 1;
    for (int i = 0; i < 4; ++i) {
        int xx = (x + dx[i] + n) % n, yy = (y + dy[i] + m) % m;
        int lxx = lx + dx[i], lyy = ly + dy[i];
        if (!g[xx][yy]) {
            if (vis[xx][yy][1] != lxx || vis[xx][yy][2] != lyy ||
                !vis[xx][yy][0])
                dfs(xx, yy, lxx, lyy);
        }
    }
}
int main() {
    while (cin >> n >> m) {
        flag = false;
        memset(g, 0, sizeof(g));
        memset(vis, 0, sizeof(vis));

        int g_x, g_y;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < m; ++j) {
                char ch;
                cin >> ch;
                if (ch == '#')
                    g[i][j] = 1;
                if (ch == 'S')
                    g_x = i, g_y = j;
            }
        dfs(g_x, g_y, g_x, g_y);
        cout << (flag ? "Yes" : "No") << endl;
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

> [!NOTE] **[Luogu 又是毕业季II](https://www.luogu.com.cn/problem/P1414)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 先求每个数字的因子并统计次数，然后根据次数去更新 r 最后输出即可
> 
> **注意次数少的可以使用右侧次数多的**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 1e4 + 10, M = 1e6 + 10;

int n;
int a[N], c[M], r[N];
unordered_map<int, int> cnt;

int main() {
    cin >> n;
    for (int i = 1; i <= n; ++ i )
        cin >> a[i], cnt[a[i]] ++ ;
    
    for (auto [k, v] : cnt)
        for (int i = 1; i <= k / i; ++ i )
            if (k % i == 0) {
                c[i] += v ;
                if (k / i != i)
                    c[k / i] += v ;
            }
    
    for (int i = 1; i < M; ++ i )
        r[c[i]] = max(r[c[i]], i);
    
    // ATTENTION 次数多于本个的 本个一定可以用
    for (int i = n - 1; i >= 1; -- i )
        r[i] = max(r[i], r[i + 1]);
    
    for (int i = 1; i <= n; ++ i )
        cout << r[i] << endl;
    
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

> [!NOTE] **[Luogu [AHOI2005]约数研究](https://www.luogu.com.cn/problem/P1403)**
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

// 同样的题意 数据范围更大：https://www.luogu.com.cn/problem/SP26073
//
// 本题当然可以考虑遍历解决：
// [1, n] 中有i的个数是 n/i
// 故所求即 n/1 + n/2 + ... + n/n
// res += n / i 即可
//
// 优化: n/i 相同的连续i  考虑分块
// 再把 n/i = n/j 的连续部分一次性算掉

int main() {
    int n, res;
    cin >> n;
    for (int i = 1, j; i <= n; i = j + 1) {
        // 最后的位值
        j = n / (n / i);
        // 计算结果相同(均为n/i)的有 j-i+1 个
        res += (n / i) * (j - i + 1);
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

> [!NOTE] **[Luogu 小猪佩奇爬树](https://www.luogu.com.cn/problem/P5588)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> `不管本节点是何类型先累计 最终判断再决定使用哪个` 思维
> 
> 乘法原理优化
> 
> $O(n)$ 思路重复做

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

// 易分析得出：
// 只有所有的同颜色点在一条链上（分两种情况）才对结果有贡献
// 1. 链只有一个点
// 2. 链有多个点 但必须有且只有2个端点
// https://www.cnblogs.com/Ning-H/p/11670828.html

using LL = long long;
const int N = 1e6 + 10, M = N << 1;

int n;
int w[N], tot[N], nos[N];
int h[N], e[M], ne[M], idx;
int sz[N], cnt[N], enos[N]; // used in dfs
LL res1[N], res2[N];

void init() {
    memset(h, -1, sizeof h);
    idx = 0;
    
    memset(sz, 0, sizeof sz);
    memset(cnt, 0, sizeof cnt);
    memset(enos, 0, sizeof enos);
    memset(res1, 0, sizeof res1);
    memset(res2, 0, sizeof res2);
}

void add(int a, int b) {
    e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
}

// 核心在于如何【判断端点】
// 1. 对于每一个点u 检查其是否只有一条边所连的子树
//    中有与其相同颜色的点 如果是就是一个端点【对应某节点是最上方端点的情况】
// 2. 如果左右子树都没有 而本节点既不是访问到的第一个点也不是最后一个点
//    就flag ++ 
void dfs(int u, int fa) {
    // c 颜色, k 到目前为止有多少个该颜色点
    int c = w[u], k = cnt[c];
    // flag 代表当前节点下子树颜色也为 c 的个数
    // 具体来说 【本个节点为根时，有多少个子树包含颜色为c的点】
    int flag = 0, pos = 0;
    
    sz[u] = 1;
    for (int i = h[u]; ~i; i = ne[i]) {
        int j = e[i];
        if (j != fa) {
            int last_cnt = cnt[c];
            dfs(j, u);
            // 暂时不管本个节点是否合法
            // 尽管累积所有子树的数量乘积
            // [乘法原理]
            res1[u] += (LL)sz[u] * sz[j];
            sz[u] += sz[j];
            // 更新flag pos
            if (last_cnt != cnt[c])
                flag ++ , pos = j;
        }
    }
    // 累积向上的数量乘积
    res1[u] += (LL)sz[u] * (n - sz[u]);

    // 我把这个放前面 感觉更合理些
    // 加入本个节点的节点信息
    cnt[c] ++ ;

    // ATTENTION
    // 另一种可能: 1. 进入当前节点时 cnt[c] 已经有值(不是第一个点)
    //           或
    //           2. 当前节点不为当前颜色的最后一个节点
    // 也要使 flag++。
    // 【为什么这里要 ++ ？】
    // 【端点判断【2】】
    // TODO 修改这里
    if (k || cnt[c] != tot[c])
        flag ++ ;
    
    // ATTENTION
    // flag 为 1 说明[有可能]是一个端点
    // 【端点判断【1】】
    if (flag == 1) {
        if (!enos[c])
            nos[c] = u;
        else {
            // 不管是否有多于两个端点的
            // 尽管按两个端点来算先
            int p = pos ? n - sz[pos] : sz[u];
            res2[c] = (LL)sz[nos[c]] * p;
        }
        enos[c] ++ ;
    }
}

int main() {
    init();
    cin >> n;
    for (int i = 1; i <= n; ++ i ) {
        cin >> w[i];
    
        tot[w[i]] ++ ;
        nos[w[i]] = i;  // 某个颜色对应的点为 I 
    }
    for (int i = 0; i < n - 1; ++ i ) {
        int a, b;
        cin >> a >> b;
        add(a, b), add(b, a);
    }
    
    dfs(1, -1);
    
    for (int i = 1; i <= n; ++ i ) {
        if (tot[i] == 0)
            cout << (LL)n * (n - 1) / 2 << endl;
        else if (tot[i] == 1)
            cout << res1[nos[i]] << endl;
        else if (enos[i] == 2)
            cout << res2[i] << endl;
        else
            cout << 0 << endl;
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