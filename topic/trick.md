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