## 习题

### STL

> [!NOTE] **[LeetCode 537. 复数乘法](https://leetcode-cn.com/problems/complex-number-multiplication/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> `sscanf`

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    string complexNumberMultiply(string a, string b) {
        int x1, y1, x2, y2;
        sscanf(a.c_str(), "%d+%di", &x1, &y1);
        sscanf(b.c_str(), "%d+%di", &x2, &y2);
        return to_string(x1 * x2 - y1 * y2) + "+" + to_string(x1 * y2 + x2 * y1) + "i";
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

> [!NOTE] **[LeetCode 539. 最小时间差](https://leetcode-cn.com/problems/minimum-time-difference/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 标准**

```cpp
class Solution {
public:
    int findMinDifference(vector<string> & timePoints) {
        int res = INT_MAX;
        vector<int> q;
        for (auto t : timePoints) {
            int h, m;
            sscanf(t.c_str(), "%d:%d", &h, &m);
            q.push_back(h * 60 + m);
        }
        sort(q.begin(), q.end());
        for (int i = 1; i < q.size(); ++ i ) res = min(res, q[i] - q[i - 1]);
        res = min(res, 1440 - q.back() + q[0]);
        return res;
    }
};
```

##### **C++**

```cpp
class Solution {
public:
    int get(string s) {
        int h, m;
        sscanf(s.c_str(), "%d:%d", &h, &m);
        return h * 60 + m;
    }
    int findMinDifference(vector<string>& timePoints) {
        set<string> S;
        int res = INT_MAX;
        for (auto p : timePoints) {
            if (!S.empty()) {
                auto r = S.lower_bound(p);
                int pv = get(p);
                if (r != S.end()) {
                    int rv = get(*r);
                    int dis = abs(pv - rv);
                    //cout << "pv = " << pv << " rv = " << rv << endl;
                    res = min(res, min(dis, 1440 - dis));
                } else {
                    int rv = get(*S.begin());
                    int dis = abs(pv - rv);
                    //cout << "pv = " << pv << " rv = " << rv << endl;
                    res = min(res, min(dis, 1440 - dis));
                }
                if (r != S.begin()) {
                    int lv = get(*(prev(r)));
                    int dis = abs(pv - lv);
                    //cout << "pv = " << pv << " lv = " << lv << endl;
                    res = min(res, min(dis, 1440 - dis));
                } else {
                    int lv = get(*S.rbegin());
                    int dis = abs(pv - lv);
                    //cout << "pv = " << pv << " lv = " << lv << endl;
                    res = min(res, min(dis, 1440 - dis));
                }

            }
            
            S.insert(p);
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

> [!NOTE] **[LeetCode 609. 在系统中查找重复文件](https://leetcode-cn.com/problems/find-duplicate-file-in-system/)**
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
    vector<vector<string>> findDuplicate(vector<string>& paths) {
        unordered_map<string, vector<string>> hash;
        for (auto & path : paths) {
            stringstream ssin(path);
            string p, file, name, content;
            ssin >> p;
            while (ssin >> file) {
                int x = file.find('('), y = file.find(')');
                name = file.substr(0, x), content = file.substr(x + 1, y - x - 1);
                hash[content].push_back(p + '/' + name);
            }
        }
        vector<vector<string>> res;
        for (auto & [k, v] : hash)
            if (v.size() > 1)
                res.push_back(v);
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

### 补全思维

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

> [!NOTE] **[LeetCode 400. 第N个数字](https://leetcode-cn.com/problems/nth-digit/)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> [espressif-2021](https://github.com/OpenKikCoc/LeetCode/blob/master/Contest/2021-08-21_espressif-2021/README.md) 学到的新思路
>
> > 思考 0
>
> 经典 数位dp 重复做
>
> 同剑指 offer [44. 数字序列中某一位的数字](https://leetcode-cn.com/problems/shu-zi-xu-lie-zhong-mou-yi-wei-de-shu-zi-lcof/)

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using LL = long long;
    int findNthDigit(int n) {
        LL k = n;
        for (LL i = 1; ; ++ i )
            if (i * pow(10, i) > k)
                return to_string(k / i)[k % i] - '0';
            else
                k += pow(10, i);
        return -1;
    }
};
```

##### **C++**

```cpp
class Solution {
public:

    int findNthDigit(int n) {
        long long k = 1, t = 9, s = 1;
        while (n > k * t) {
            n -= k * t;
            k ++, t *= 10, s *= 10;
        }
        // s + [n/k] 向上取整
        s += (n + k - 1) / k - 1;
        n = n % k ? n % k : k;
        return to_string(s)[n - 1] - '0';
    }

    int findNthDigit_2(int n) {
        int base = 1;
        while (n > 9 * pow(10, base - 1) * base) {
            n -= 9 * pow(10, base - 1) * base;
            ++ base ;
        }
        int value = pow(10, base - 1) + n / base;
        int mod = n % base;
        if (mod) return value / (int)pow(10, base - mod) % 10;
        return (value - 1) % 10;
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

> [!NOTE] **[LeetCode 1366. 通过投票对团队排名](https://leetcode-cn.com/problems/rank-teams-by-votes/)**
> 
> 题意: 
> 
> 参赛团队的排名次序依照其所获「排位第一」的票的多少决定。如果存在多个团队并列的情况，将继续考虑其「排位第二」的票的数量。
> 
> 以此类推，直到不再存在并列的情况。 如果在考虑完所有投票情况后仍然出现并列现象，则根据团队字母的字母顺序进行排名。

> [!TIP] **思路**
> 
> **巧妙的补全一个字符维度**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 裸排序**

```cpp
class Solution {
public:
    struct node {
        int cnt = 0, c = 0, n;
        unordered_map<int, int> v;
        bool operator<(const node& n) const {
            int sz = this->n, a, b;
            auto mpa = this->v, mpb = n.v;
            for (int i = 0; i < sz; ++i) {
                a = mpa[i], b = mpb[i];
                if (a != b) return a > b;
            }
            return c < n.c;
            // return rank < n.rank;
        }
    };
    string rankTeams(vector<string>& votes) {
        int n = votes.size();
        int m = votes[0].size();
        vector<node> v(26);
        for (int i = 0; i < 26; ++i) v[i].c = i, v[i].n = m;

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                ++v[votes[i][j] - 'A'].v[j];
                ++v[votes[i][j] - 'A'].cnt;
            }
        }
        sort(v.begin(), v.end());
        string res;
        for (int i = 0; i < 26; ++i) {
            if (v[i].cnt)
                res.push_back(v[i].c + 'A');
            else
                break;
        }

        return res;
    }
};
```

##### **C++ 补全**

```cpp
class Solution {
public:
    string rankTeams(vector<string>& votes) {
        string res;
        vector<vector<int>> dw(27, vector<int>(27, 0));
        for (auto p : votes) {
            for (int i = 0; i < p.length(); ++i) {
                dw[p[i] - 'A'][i]++;
                dw[p[i] - 'A'][26] = 26 - (p[i] - 'A');
            }
        }
        sort(dw.begin(), dw.end(), greater<vector<int>>());
        for (int i = 0; i < dw.size(); ++i) {
            if (dw[i][26]) res.push_back(26 - dw[i][26] + 'A');
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

### 思维 ==> TODO 细分

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

> [!NOTE] **[LeetCode 423. 从英文中重建数字](https://leetcode-cn.com/problems/reconstruct-original-digits-from-english/)**
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
/*
        “z” 只在 “zero” 中出现。同时带走o
        “g” 只在 “eight” 中出现, 同时把h给带走了
        “h” 在剩下的单词中只有"three"有
        “w” 只在 “two” 中出现。同时把o带走
        “x” 只在 “six” 中出现
        “u” 只在 “four” 中出现。同时把f，o带走
        "f" 在剩下的单词中只有"five"有
        "o" 在剩下的单词里只有"one"有
        “s" 在剩下的单词里只有"seven"有
        "n" 在剩下的单词里只有"nine“有
*/
    string originalDigits(string s) {
        string name[] = {
            "zero", "one", "two", "three", "four", "five",
            "six", "seven", "eight", "nine"
        };
        int ord[] = {0, 8, 3, 2, 6, 4, 5, 1, 7, 9};
        unordered_map<char, int> cnt;
        for (auto c: s) cnt[c] ++ ;
        string res;
        for (int x: ord) {
            while (true) {
                bool flag = true;
                for (auto c: name[x])
                    if (!cnt[c]) {
                        flag = false;
                        break;
                    }
                    if (flag) {
                        res += to_string(x);
                        for (auto c: name[x]) cnt[c] -- ;
                    }
                    else break;
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

> [!NOTE] **[LeetCode 1419. 数青蛙](https://leetcode-cn.com/problems/minimum-number-of-frogs-croaking/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> "croak"合法串中 尽可能最少的青蛙的个数

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int minNumberOfFrogs(string croakOfFrogs) {
        int n = croakOfFrogs.size();
        if (n % 5) return -1;
        int res = 0;
        vector<int> hasNum(5, 0);
        // 叫了c的 r的 a的 k的 必须时刻有顺序递减 >= 否则不合法
        // 合法的情况下 c的数量就是当前最少的 统计c
        for (int i = 0; i < n; ++i) {
            if (croakOfFrogs[i] == 'c')
                ++hasNum[0];
            else if (croakOfFrogs[i] == 'r')
                ++hasNum[1];
            else if (croakOfFrogs[i] == 'o')
                ++hasNum[2];
            else if (croakOfFrogs[i] == 'a')
                ++hasNum[3];
            else if (croakOfFrogs[i] == 'k')
                ++hasNum[4];

            if (hasNum[0] < hasNum[1] || hasNum[1] < hasNum[2] ||
                hasNum[2] < hasNum[3] || hasNum[3] < hasNum[4])
                return -1;
            res = max(res, hasNum[0]);
            if (hasNum[4])
                --hasNum[0], --hasNum[1], --hasNum[2], --hasNum[3], --hasNum[4];
        }
        if (hasNum[0] || hasNum[1] || hasNum[2] || hasNum[3] || hasNum[4])
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

> [!NOTE] **[LeetCode 73. 矩阵置零](https://leetcode-cn.com/problems/set-matrix-zeroes/)**
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
    void setZeroes(vector<vector<int>>& matrix) {
        if(matrix.empty()) return;
        int m = matrix.size(), n = matrix[0].size();
      
        bool firstrow = false, firstcollumn = false;
        for (int i = 0; i < n; ++ i )
            if (matrix[0][i] == 0) firstrow = true;
        for (int i = 0; i < m; ++ i )
            if (matrix[i][0] == 0) firstcollumn = true;
        
        for (int i = 1; i < m; ++ i )
            for (int j = 1; j < n; ++ j )
                if (matrix[i][j] == 0) {
                    matrix[i][0] = 0;
                    matrix[0][j] = 0;
                }
            
        for (int i = 1; i < m; ++ i )
            for (int j = 1; j < n; ++ j )
                if(matrix[i][0] == 0 || matrix[0][j] == 0) matrix[i][j] = 0;
            
        if (firstrow)
            for(int i = 0; i < n; ++ i ) matrix[0][i] = 0;
        if (firstcollumn)
            for(int i = 0; i < m; ++ i ) matrix[i][0] = 0;
    }
};
```

##### **Python**

```python
# 暴力循环，有重复计算；只需要统计出矩阵的每一行或者每一列是否有0
# 1. 用两个变量记录第一行和第一列是否有0
# 2. 遍历整个矩阵，用矩阵的第一行和第一列记录
# 3. 把含有0的行和列都置为0

class Solution:
    def setZeroes(self, arr: List[List[int]]) -> None:
        n, m = len(arr), len(arr[0])
        r0, c0 = 1, 1
        for i in range(n):
            if not arr[i][0]:c0 = 0
        for i in range(m):
            if not arr[0][i]:r0 = 0
        for i in range(1, n):
            for j in range(1, m):
                if not arr[i][j]:
                    arr[i][0] = 0
                    arr[0][j] = 0
        # 开始变换 0 ，遍历第1行 和 第1列暂存的数据            
        for i in range(1, n):
            if not arr[i][0]:
                for j in range(1, m):
                    arr[i][j] = 0 
        for i in range(1, m):
            if not arr[0][i]:
                for j in range(1, n):
                    arr[j][i] = 0 
        # 判断 存第一行 第一列的数据           
        if not r0:
            for i in range(m):
                arr[0][i] = 0 
        if not c0:
            for i in range(n):
                arr[i][0] =  0
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 128. 最长连续序列](https://leetcode-cn.com/problems/longest-consecutive-sequence/)**
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
    int longestConsecutive(vector<int>& nums) {
        unordered_set<int> S;
        for (auto x: nums) S.insert(x);

        int res = 0;
        for (auto x: nums) {
            if (S.count(x) && !S.count(x - 1)) {
                int y = x;
                S.erase(x);
                while (S.count(y + 1)) {
                    y ++ ;
                    S.erase(y);
                }
                res = max(res, y - x + 1);
            }
        }

        return res;
    }
};
```

##### **C++ 其他思路**

```cpp
class Solution {
public:
    int longestConsecutive(vector<int>& nums) {
        unordered_map<int, int> tr_l, tr_r;
        int res = 0;
        for (auto & x : nums) {
            int l = tr_r[x - 1], r = tr_l[x + 1];
            tr_l[x - l] = max(tr_l[x - l], l + r + 1);
            tr_r[x + r] = max(tr_r[x + r], l + r + 1);
            res = max(res, l + r + 1);
        }
        return res;
    }

    int longestConsecutive2(vector<int>& nums) {
        int n = nums.size();
        unordered_map<int, bool> m;
        for (auto v : nums) m[v] = true;
        int res = 0;
        for (auto v : nums) {
            if (m[v - 1]) continue;
            int cnt = 1;
            while (m[ ++ v]) ++ cnt ;
            res = max(res, cnt);
        }
        return res;
    }
};
```

##### **Python**

```python
# 1. 把所有的数都存到哈希中； 遍历哈希表中的元素，因为要找连续的数字序列，因此可以通过向后枚举相邻的数字（即不断加一），判断后面一个数字是否在哈希表中即可。
# 2. 为了避免重复枚举序列，因此只对序列的起始数字向后枚举，（例如[1,2,3,4]，只对1枚举，2，3，4时跳过），因此需要判断一下是否是序列的起始数字（即判断一下n-1是否在哈希表中）。
# 3. 由于用哈希 或者用集合时，已经去过重了，而且遍历的是哈希里的key, 所以每次处理完后不需要另外删除掉当前数

class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        my_cnt = collections.Counter(nums)
        res = 0
        for x in my_cnt:
            if x in my_cnt and x - 1 not in my_cnt:
                y = x
                while y + 1 in my_cnt:
                    y += 1
                res = max(res, y - x + 1)
        return res

# class Solution:
#     def longestConsecutive(self, nums: List[int]) -> int:
#         s = set(nums)
#         res = 0

#         for x in s:
#             if x in s and x - 1 not in s:
#                 y = x
#                 while y + 1 in s:
#                     y += 1
#                 res = max(res, y - x + 1)
#         return res

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 258. 各位相加](https://leetcode-cn.com/problems/add-digits/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> `x * 100 + y * 10 + z = x * 99 + y * 9 + x + y + z`
> 
> 1. 能够被 9 整除的整数，各位上的数字加起来也必然能被 9 整除
> 
>    所以，连续累加起来，最终必然就是 9 。
> 
> 2. 不能被 9 整除的整数，各位上的数字加起来，结果对 9 取模，和初始数对 9 取摸，是一样的
> 
>    所以，连续累加起来，最终必然就是初始数对 9 取摸。

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int addDigits(int num) {
        if (!num) return 0;
        if (num % 9) return num % 9;
        return 9;
    }

    int addDigits(int num) {
        while (num >= 10) {
            int t = 0;
            while (num) {
                t += num % 10;
                num /= 10;
            }
            num = t;
        }
        return num;
    }
};
```

##### **Python**

```python
"""
1. 能够被 9 整除的整数，各位上的数字加起来也必然能被 9 整除

   所以，连续累加起来，最终必然就是 9 。

2. 不能被 9 整除的整数，各位上的数字加起来，结果对 9 取模，和初始数对 9 取摸，是一样的

   所以，连续累加起来，最终必然就是初始数对 9 取摸。
"""

class Solution:
    def addDigits(self, num: int) -> int:
        # num=str(num)
        # while(len(num)>1):
        #     tmp=0
        #     for a in num:
        #         tmp+=int(a)
        #     tmp=str(tmp)
        #     num=tmp
        # return int(num)
        
        #法二
        # while num>=10:
        #     tot=0
        #     while num:
        #         tot+=num%10
        #         num=num//10
        #     num=tot    
        # return num

        if num<9:
            return num
        elif num%9:
            return num%9
        else:
            return 9
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 334. 递增的三元子序列](https://leetcode-cn.com/problems/increasing-triplet-subsequence/)**
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
    bool increasingTriplet(vector<int>& nums) {
        int v1 = INT_MAX, v2 = INT_MAX;
        for (int v : nums) {
            // 已满足三个数
            if (v > v2) return true;
            // 更新两个数
            if (v1 != INT_MAX && v > v1 && v < v2) v2 = v;
            // 更新一个数
            if (v < v1) v1 = v;
        }
        return false;
    }
};
```

##### **C++**

```cpp
// yxc
class Solution {
public:
    bool increasingTriplet(vector<int>& nums) {
        vector<int> q(2, INT_MAX);
        for (auto a: nums) {
            int k = 2;
            while (k > 0 && q[k - 1] >= a) k -- ;
            if (k == 2) return true;
            q[k] = a;
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

> [!NOTE] **[LeetCode 414. 第三大的数](https://leetcode-cn.com/problems/third-maximum-number/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 实现比较 trick

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int thirdMax(vector<int>& nums) {
        long long INF = 1e10, a = -INF, b = -INF, c = -INF, s = 0;
        for (auto x: nums) {
            if (x > a) s ++, c = b, b = a, a = x;
            else if (x < a && x > b) s ++, c = b, b = x;
            else if (x < b && x > c) s ++, c = x;
        }
        if (s < 3) return a;
        return c;
    }
};
```

##### **Python**

```python
"""
模拟遍历：每次保存更新最大值，次大值，第三大值
"""

class Solution:
    def thirdMax(self, nums: List[int]) -> int:
        a=-1e20;b=-1e20;c=-1e20;s=0
        for x in nums:
            if x>a: c=b;b=a;a=x;s+=1
            elif x<a and x>b: c=b;b=x;s+=1
            elif x<b and x>c: c=x;s+=1
        if s<3: return a
        return c
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 447. 回旋镖的数量](https://leetcode-cn.com/problems/number-of-boomerangs/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 对 On^3 的优化

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 对 On^3 的优化，枚举 i 检查多少个相同距离的点
    int numberOfBoomerangs(vector<vector<int>>& points) {
        int res = 0;
        for (int i = 0; i < points.size(); ++ i ) {
            unordered_map<int, int> cnt;
            for (int j = 0; j < points.size(); ++ j ) {
                if (i != j) {
                    int dx = points[i][0] - points[j][0], dy = points[i][1] - points[j][1];
                    int dist = dx * dx + dy * dy;
                    ++ cnt[dist];
                }
            }
            for (auto [d, c] : cnt) res += c * (c - 1);
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

> [!NOTE] **[LeetCode 479. 最大回文数乘积](https://leetcode-cn.com/problems/largest-palindrome-product/)**
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
    typedef long long LL;
    int largestPalindrome(int n) {
        if (n == 1) return 9;
        int maxv = pow(10, n) - 1;
        for (int i = maxv; ; -- i ) {
            auto s = to_string(i);
            auto t = s;
            reverse(t.begin(), t.end());
            // 根据某个 n 位数 s , 构造可能的回文数答案: num
            auto num = stoll(s + t);
            for (LL j = maxv; j * j >= num; -- j )
                if (num % j == 0) return num % 1337;
        }
        return 0;
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

> [!NOTE] **[LeetCode 517. 超级洗衣机](https://leetcode-cn.com/problems/super-washing-machines/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 证明和推导过程 证明可取下界
> 
> 思维 重复

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 由于每次操作每台洗衣机只能选择向左或者向右运送一件衣服，且多个洗衣机可以并行同时运送，
    // 故必定存在一个洗衣机，它运送的衣服数量等于答案。
    // 遍历取最大的下界 证明参考acwing
    int findMinMoves(vector<int>& machines) {
        int n = machines.size(), sum = 0;
        for (auto v : machines) sum += v;
        if (sum % n) return -1;
        int avg = sum / n, left = 0, right = sum;
        int res = 0;
        for (int i = 0; i < n; ++ i ) {
            right -= machines[i];
            if (i) left += machines[i - 1];
            int l = max(avg * i - left, 0), r = max(avg * (n - i - 1) - right, 0);
            res = max(res, l + r);
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

> [!NOTE] **[LeetCode 564. 寻找最近的回文数](https://leetcode-cn.com/problems/find-the-closest-palindrome/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> trick 思维 实现

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // n 长度不超过18 则数字范围不超过long long
    typedef long long LL;
    string nearestPalindromic(string n) {
        int len = n.size();
        set<LL> S;
        // 大小边界
        S.insert((LL)pow(10, len - 1) - 1);
        S.insert((LL)pow(10, len) + 1);
        // 考虑复制前半部分
        LL m = stoll(n.substr(0, (len + 1) / 2));
        for (LL i = m - 1; i <= m + 1; ++ i ) {
            string a = to_string(i), b = a;
            reverse(b.begin(), b.end());
            if (len % 2) S.insert(stoll(a + b.substr(1)));
            else S.insert(stoll(a + b));
        }
        LL k = stoll(n);
        S.erase(k);
        LL res = 2e18;
        for (auto x : S)
            if (abs(x - k) < abs(res - k))
                res = x;
        return to_string(res);
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

> [!NOTE] **[LeetCode 581. 最短无序连续子数组](https://leetcode-cn.com/problems/shortest-unsorted-continuous-subarray/)**
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
    // 基于 若有序 当前nums[i]必然是前面第m大 后面同理
    int findUnsortedSubarray(vector<int>& nums) {
        int m = nums[0], n = nums.back(), l = -1, r = -2;
        int len = nums.size();
        for (int i = 1; i < len; ++ i ) {
            m = max(m, nums[i]);
            n = min(n, nums[len - 1 - i]);
            if (m != nums[i]) r = i;
            if (n != nums[len - 1 - i]) l = len - 1 - i;
        }
        return r - l + 1;
    }
}
```

##### **C++ 2**

题解2：

整体思路和题解1其实是类似的，找到两段已经排好序的长度。

1. 我们先使用一遍扫描找到左边保持升序的最后一个点的位置left,和从右向左看保持降序的最后一个点的位置right。
2. 如果已经这时候left == right说明已经排好序了，无需调整。
3. 接下来我们从left + 1的位置向右扫描，如果遇到有比nums[left]小的元素，说明最起码left不在正确位置上，那么left --。
4. 同样的，我们从right - 1的位置向左扫描，如果遇到有比nums[right]大的元素，说明最起码nums[right]不在正确的位置上，right ++。
5. 最后，left和right之间的元素就是需要重新排序的元素，长度为right - left - 1。
6. 
```cpp
// yxc 思路
class Solution {
public:
    int findUnsortedSubarray(vector<int>& nums) {
        int l = 0, r = nums.size() - 1;
        while (l + 1 < nums.size() && nums[l + 1] >= nums[l]) l ++ ;
        if (l == r) return 0;
        while (r - 1 >= 0 && nums[r - 1] <= nums[r]) r -- ;
        for (int i = l + 1; i < nums.size(); i ++ )
            while (l >= 0 && nums[l] > nums[i])
                l -- ;
        for (int i = r - 1; i >= 0; i -- )
            while (r < nums.size() && nums[r] < nums[i])
                r ++ ;
        return r - l - 1;

    }
};
```

##### **Python**

```python
"""
题解1:排序，然后分别找到两端已经排好序的长度。中间那部分就是需要重新排列的长度。时间复杂度O(nlogn)，空间复杂度O(n)。
"""

class Solution:
    def findUnsortedSubarray(self, nums: List[int]) -> int:
        a = sorted(nums)
        n = len(nums)
        l, r = 0, n - 1
        while l < n and nums[l] == a[l]:
            l += 1
        while r >= l and nums[r] == a[r]:
            r -= 1
        return r - l + 1



"""
整体思路和题解1其实是类似的，找到两段已经排好序的长度。
1. 我们先使用一遍扫描找到左边保持升序的最后一个点的位置left,和从右向左看保持降序的最后一个点的位置right。
2. 如果已经这时候left == right说明已经排好序了，无需调整。
3. 接下来我们从left + 1的位置向右扫描，如果遇到有比nums[left]小的元素，说明最起码left不在正确位置上，那么left --。
4. 同样的，我们从right - 1的位置向左扫描，如果遇到有比nums[right]大的元素，说明最起码nums[right]不在正确的位置上，right ++。
5. 最后，left和right之间的元素就是需要重新排序的元素，长度为right - left - 1。
"""
class Solution(object):
    def findUnsortedSubarray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """

        l = 0
        r = len(nums) - 1
        # 定义初始左边界，[0-l]都是递增的
        while l + 1 < len(nums) and nums[l + 1] >= nums[l]:
            l += 1

        # 说明当前数组是纯递增的
        if l == r:
            return 0

        # 定义初始右边界，[r, len(nums)-1]都是递增的 
        while r - 1 >= 0 and nums[r - 1] <= nums[r]:
            r -= 1

        # 重新定义左右有序边界
        # 找到真正的左边界，因为左边界右边的所有数，都应该比左边界右边的所有数要小
        for i in range(l + 1, len(nums)):
            while l >= 0 and nums[l] > nums[i]:
                l -= 1

        # 找到真正的右边界，因为右边界左边的所有数，都应该比右边界右边的所有数要大
        for i in range(r-1, -1, -1):
            while r < len(nums) and nums[r] < nums[i]:
                r += 1

        # 最后得到的[0-l-r-len(nums)]
        # (l,r)这个区间内才是真正需要排序的，[0-l],[r-len(nums)]都是已经从小到大排好的
        return r - l - 1 
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 621. 任务调度器](https://leetcode-cn.com/problems/task-scheduler/)**
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
    int leastInterval(vector<char>& tasks, int n) {
        vector<int> cnt(26);
        int mf = 0;
        for (auto c : tasks) ++ cnt[c - 'A'], mf = max(mf, cnt[c - 'A']);
        int res = (mf - 1) * (n + 1);
        for (auto v : cnt) if (v == mf) ++ res;
        return max(res, (int)tasks.size());
    }
};
```

##### **Python**

```python
"""
由于相同任务之间需要等待n时间，所以一个比较自然的想法就是找到出现最多的任务，如A
然后将其余任务安排在两个A之间的n个间隔内；
当然有可能出现最多的任务有多个，如A，B均出现了100次；
另外就是最后一组任务不需要在完成后等待；
于是有(maxCount - 1) * (n+1) + count

但是，如果任务的重复较少但每个任务数量又很多，且同任务间的间隔很低，这时候以maxCount - 1作为底数就有很多任务无法纳入其间隔内
这时候就通过len(tasks)//(n+1) * (n+1)来替代最多任务出现的次数；
对于最后一组任务，需要加上len(tasks)%(n+1)
以上其实就是len(tasks)


"""

class Solution:
    def leastInterval(self, tasks: List[str], n: int) -> int:
        task_dict = collections.Counter(tasks)

        # 最多任务出现的次数
        maxCount = max(task_dict.values())
        # 最多任务共有多少个
        count = 0
        for val in task_dict.values():
            if val == maxCount:
                count += 1

        return max((maxCount - 1) * (n+1) + count, len(tasks))
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 662. 二叉树最大宽度](https://leetcode-cn.com/problems/maximum-width-of-binary-tree/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 如果按照一般树编号 则爆 long long 需要每层都从1开始编号

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    int widthOfBinaryTree(TreeNode* root) {
        if (!root) return 0;
        queue<pair<TreeNode*, int>> q;
        q.push({root, 1});
        int res = 1;
        while (q.size()) {
            int sz = q.size();
            // 最左侧节点 l
            int l = q.front().second, r;

            while (sz -- ) {
                auto [node, rv] = q.front(); q.pop();
                r = rv;
                int id = rv - l + 1;
                
                if (node->left) q.push({node->left, id * 2});
                if (node->right) q.push({node->right, id * 2 + 1});
            }
            res = max(res, r - l + 1);
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

> [!NOTE] **[LeetCode 665. 非递减数列](https://leetcode-cn.com/problems/non-decreasing-array/)**
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
    bool check(vector<int>& nums) {
        for (int i = 1; i < nums.size(); i ++ )
            if (nums[i] < nums[i - 1])
                return false;
        return true;
    }

    bool checkPossibility(vector<int>& nums) {
        for (int i = 1; i < nums.size(); i ++ )
            if (nums[i] < nums[i - 1]) {
                int a = nums[i - 1], b = nums[i];
                nums[i - 1] = nums[i] = a;
                if (check(nums)) return true;
                nums[i - 1] = nums[i] = b;
                if (check(nums)) return true;
                return false;
            }
        return true;
    }
};
```

##### **C++ 早期代码**

```cpp
class Solution {
public:
    bool checkPossibility(vector<int>& nums) {
        int len = nums.size();
        if (len <= 1) return true;
        bool change = false;
        for (int i = 1; i < len; ++ i ) {
            if (nums[i - 1] > nums[i]) {
                // 应该修改前面的数
                // for [3,4,2,3]
                // for [2,3,3,2,4]
                // for [4,2,3]
                if (!change) {
                    if(i >= 2 && nums[i - 2] > nums[i]) nums[i] = nums[i - 1];
                    else nums[i - 1] = nums[i];
                    change = true;
                } else return false;
            }
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

> [!NOTE] **[LeetCode 717. 1比特与2比特字符](https://leetcode-cn.com/problems/1-bit-and-2-bit-characters/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> trick

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    bool isOneBitCharacter(vector<int>& bits) {
        for (int i = 0; i < bits.size(); i ++ ) {
            if (i == bits.size() - 1 && !bits[i]) return true;
            if (bits[i]) i ++ ;
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

> [!NOTE] **[LeetCode 738. 单调递增的数字](https://leetcode-cn.com/problems/monotone-increasing-digits/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 思维题 重复

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int monotoneIncreasingDigits(int N) {
        string ns = to_string(N);
        int n = ns.size();
        int i = 1;
        while (i < n && ns[i - 1] <= ns[i]) ++ i ;
        if (i < n) {
            // now ns[i - 1] > ns[i]
            while (i > 0 && ns[i - 1] > ns[i]) -- ns[i - 1], -- i ;
            for (int j = i + 1; j < n; ++ j ) ns[j] = '9';
        }
        return stoi(ns);
    }
};
```

##### **C++**

```cpp
class Solution {
public:
    int monotoneIncreasingDigits(int N) {
        auto str = to_string(N);
        int k = 0;
        while (k + 1 < str.size() && str[k] <= str[k + 1]) k ++ ;
        if (k == str.size() - 1) return N;
        while (k && str[k - 1] == str[k]) k -- ;
        str[k] -- ;
        for (int i = k + 1; i < str.size(); i ++ ) str[i] = '9';
        return stoi(str);
    }
};
```

##### **Python**

```python
class Solution:
    def monotoneIncreasingDigits(self, N: int) -> int:
        s = list(str(N))  # 踩坑，要转化为string类型的列表
        n = len(s)
        k = 0 
        while k + 1 < n and s[k] <= s[k + 1]:k += 1
        if k == n - 1:return N 
        while k and s[k - 1] == s[k]:k -= 1 
        s[k] = str(int(s[k]) - 1) # 最开始相等的位置， 减去1
        for i in range(k + 1, n):  # 后面位置都变成 ‘9’
            s[i] = '9'
        return ''.join(s).lstrip('0')
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 768. 最多能完成排序的块 II](https://leetcode-cn.com/problems/max-chunks-to-make-sorted-ii/)**
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
    // 考虑: 能分尽分
    // 1. 如果当前值 `v >= lastMax` 则 v 可以单独成块
    // 2. 如果当前值 `v < lastMax` 则需往前一直找到一个块
    //      使得 `v >= someMax` 合并其间所有块
    int maxChunksToSorted(vector<int>& arr) {
        stack<int> st;
        for (auto v : arr) {
            // 维护当前块的最大值
            int t = st.empty() ? 0 : st.top();
            while (st.size() && v < st.top())
                st.pop();
            st.push(max(t, v));
        }
        return st.size();
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

> [!NOTE] **[LeetCode 769. 最多能完成排序的块](https://leetcode-cn.com/problems/max-chunks-to-make-sorted/)**
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
    int maxChunksToSorted(vector<int>& arr) {
        int mx = -1, n = arr.size();
        int cnt = 0;
        for (int i = 0; i < n; ++ i ) {
            mx = max(mx, arr[i]);
            if (mx == i)
                cnt ++ ;
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

> [!NOTE] **[LeetCode 777. 在LR字符串中交换相邻字符](https://leetcode-cn.com/problems/swap-adjacent-in-lr-string/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> trick

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    bool canTransform(string start, string end) {
        string a, b;
        for (auto c: start)
            if (c != 'X') a += c;
        for (auto c: end)
            if (c != 'X') b += c;
        if (a != b) return false;
        for (int i = 0, j = 0; i < start.size(); i ++, j ++ ) {
            while (i < start.size() && start[i] != 'L') i ++ ;
            while (j < end.size() && end[j] != 'L') j ++ ;
            if (i < j) return false;
        }
        for (int i = 0, j = 0; i < start.size(); i ++, j ++ ) {
            while (i < start.size() && start[i] != 'R') i ++ ;
            while (j < end.size() && end[j] != 'R') j ++ ;
            if (i > j) return false;
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

> [!NOTE] **[LeetCode 782. 变为棋盘](https://leetcode-cn.com/problems/transform-to-chessboard/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 数学 略麻烦
> 
> 分析易知显然只有两种棋盘，各个情况讨论可行性即可

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    const int INF = 1e8;

    int get(int a, int b) {
        if (__builtin_popcount(a) != __builtin_popcount(b)) return INF;
        return __builtin_popcount(a ^ b) / 2;
    }

    int movesToChessboard(vector<vector<int>>& board) {
        int n = board.size();
        set<int> row, col;
        for (int i = 0; i < n; ++ i ) {
            int r = 0, c = 0;
            for (int j = 0; j < n; ++ j ) {
                r |= board[i][j] << j;
                c |= board[j][i] << j;
            }
            row.insert(r), col.insert(c);
        }
        if (row.size() != 2 || col.size() != 2) return -1;
        int r1 = *row.begin(), r2 = *row.rbegin();
        int c1 = *col.begin(), c2 = *col.rbegin();
        if ((r1 ^ r2) != (1 << n) - 1 || (c1 ^ c2) != (1 << n) - 1) return -1;
        int s1 = 0;
        for (int i = 0; i < n; i += 2) s1 |= 1 << i;
        int s2 = ((1 << n) - 1) ^ s1;
        int r_cost = min(get(r1, s1), get(r1, s2));
        int c_cost = min(get(c1, s1), get(c1, s2));
        int res = r_cost + c_cost;
        return res >= INF ? -1 : res;
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

> [!NOTE] **[LeetCode 789. 逃脱阻碍者](https://leetcode-cn.com/problems/escape-the-ghosts/)**
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
    int get_dist(int x1, int y1, int x2, int y2) {
        return abs(x1 - x2) + abs(y1 - y2);
    }

    bool escapeGhosts(vector<vector<int>>& ghosts, vector<int>& target) {
        for (auto & g : ghosts)
            if (get_dist(g[0], g[1], target[0], target[1]) <= abs(target[0]) + abs(target[1]))
                return false;
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

> [!NOTE] **[LeetCode 792. 匹配子序列的单词数](https://leetcode-cn.com/problems/number-of-matching-subsequences/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> trick 类似多路归并的思路来优化匹配过程

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using PII = pair<int, int>;
    #define x first
    #define y second

    int numMatchingSubseq(string s, vector<string>& words) {
        vector<PII> ps[26];
        for (int i = 0; i < words.size(); ++ i )
            ps[words[i][0] - 'a'].push_back({i, 0});
        
        int res = 0;
        for (auto c : s) {
            // buf
            vector<PII> buf;
            for (auto & p : ps[c - 'a'])
                if (p.y + 1 == words[p.x].size()) res ++ ;
                else buf.push_back({p.x, p.y + 1});
            ps[c - 'a'].clear();
            for (auto & p : buf)
                ps[words[p.x][p.y] - 'a'].push_back(p);
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

> [!NOTE] **[LeetCode 1144. 递减元素使数组呈锯齿状](https://leetcode-cn.com/problems/decrease-elements-to-make-array-zigzag/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 注意 只递减 简单很多

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int movesToMakeZigzag(vector<int>& nums) {
        int n = nums.size();
        if (n < 2)
            return 0;
        int s1 = 0, s2 = 0;
        for (int i = 0; i < n; ++ i ) {
            int l = i > 0 ? nums[i - 1] : INT_MAX;
            int r = i < n - 1 ? nums[i + 1] : INT_MAX;
            int v = max(0, nums[i] - min(l, r) + 1);
            if (i & 1)
                s1 += v;
            else
                s2 += v;
        }
        return min(s1, s2);
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

> [!NOTE] **[LeetCode 1183. 矩阵中 1 的最大数量](https://leetcode-cn.com/problems/maximum-number-of-ones/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 计算左上角正方形的每个格子在整个矩形中有多少个等效位置，取等效位置最多的前maxOnes个即可。
> 
> > 先放好左上角，然后剩下的正方形都复制粘贴左上角的

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int maximumNumberOfOnes(int w, int h, int l, int maxOnes) {
        vector<int> ve;
        for (int i = 0; i < l; ++ i )
            for (int j = 0; j < l; ++ j ) {
                int v = 1;
                v *= (w - i - 1) / l + 1;
                v *= (h - j - 1) / l + 1;
                ve.push_back(v);
            }
        sort(ve.begin(), ve.end(), greater<int>());
        int res = 0;
        for (int i = 0; i < maxOnes; ++ i )
            res += ve[i];
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

> [!NOTE] **[LeetCode 1191. K 次串联后最大子数组之和](https://leetcode-cn.com/problems/k-concatenation-maximum-sum/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
>
> 当 k = 1 时，答案为当前数组的最大子序和（参考股票问题）
>
> 当 k >= 2 时，答案为三者的最大值：
>
> 1.  k = 1 时的答案
> 2.  最大前缀和 + 全部和 - 最小前缀和 （即：最大后缀和 + 最大前缀和）
> 3.  k = 1 时的答案 + (k - 1)*当前数组的和

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    const int mod = 1e9 + 7;
    using LL = long long;
    int kConcatenationMaxSum(vector<int>& arr, int k) {
        LL minv = 0, maxv = 0, maxd = 0, sum = 0;
        for (auto v : arr) {
            sum += v;
            if (sum < minv) minv = sum;
            if (sum > maxv) maxv = sum;
            if (sum - minv > maxd) maxd = sum - minv;
        }
        LL k1 = maxd, k2 = maxv + sum - minv, kn = maxd + (k - 1) * sum;
        return k == 1 ? k1 % mod : max({k1, k2, kn}) % mod;
    }
};
```

##### **C++ liuzhou**

```cpp
// liuzhou_101

class Solution {
public:
    const int mod = 1e9 + 7;
    using LL = long long;
    int kConcatenationMaxSum(vector<int>& arr, int k) {
        int n = arr.size();
        vector<LL> s(2 * n + 1);
        LL minv = 0, ret1 = 0, ret2 = 0;
        for (int i = 1; i <= 2 * n; ++ i ) {
            s[i] = s[i - 1] + arr[(i - 1) % n];
            minv = min(minv, s[i]);
            if (i <= n)
                ret1 = max(ret1, s[i] - minv);
            ret2 = max(ret2, s[i] - minv);
        }
        
        if (k == 1) return ret1 % mod;
        if (k == 2) return ret2 % mod;
        
        LL t1 = ret1, t2 = ret2;
        t1 += LL(k - 1) * s[n];
        t2 += LL(k - 2) * s[n];
        LL res = max({ret1, ret2, t1, t2});
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

> [!NOTE] **[LeetCode 1247. 交换字符使得字符串相同](https://leetcode-cn.com/problems/minimum-swaps-to-make-strings-equal/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 规律 统计

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int minimumSwap(string s1, string s2) {
        if (s1.size() != s2.size()) return false;
        int n = s1.size();
        // 统计有多少 x-y 和 y-x
        int c1 = 0, c2 = 0;
        for (int i = 0; i < n; ++i) {
            if (s1[i] == 'x' && s2[i] == 'y')
                ++c1;
            else if (s1[i] == 'y' && s2[i] == 'x')
                ++c2;
        }
        // 对于每一对 x-y,x-y y-x,y-x 一次操作即可
        int res = c1 / 2 + c2 / 2;
        c1 %= 2, c2 %= 2;
        if (c1 + c2 == 1)
            return -1;
        else if (c1 + c1 == 2)
            res += 2;  // 一对 x-y y-x 需要两次
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

> [!NOTE] **[LeetCode 1256. 加密数字](https://leetcode-cn.com/problems/encode-number/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 规律

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

自己模拟实现如下：

主要是找到该数字所在的段，随后计算 01 值

```cpp
class Solution {
public:
    // 0位的数有1个
    // 1       2
    // 2       4
    // 3       8
    string encode(int num) {
        if (!num)
            return "";
        else if (num == 1)
            return "0";
        else if (num == 2)
            return "1";
        int b = 0, c = 1, tot = 1;
        while (num - tot >= c * 2) {
            ++b;
            c *= 2;
            tot += c;
        }
        // num 位数为 b+1
        // cout <<"num="<<num<<" b="<<b<<" c="<<c <<" tot="<<tot<<endl;
        int idx = num - tot;
        int p = 1 << b;
        string res;
        while (p) {
            if (p & idx)
                res.push_back('1');
            else
                res.push_back('0');
            p >>= 1;
        }
        return res;
    }
}
```

##### **C++ 找规律**

可以直接找规律：结果就是 原来的值+1 转换为二进制后去除最高位

```cpp
class Solution {
public:
    string encode(int num) {
        ++num;
        string res;
        while (num) {
            if (num & 1)
                res.push_back('1');
            else
                res.push_back('0');
            num >>= 1;
        }
        res.pop_back();
        reverse(res.begin(), res.end());
        return res;
    }
}
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 1328. 破坏回文串](https://leetcode-cn.com/problems/break-a-palindrome/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 删除一个字符使得原始回文串不再回文 且新字符串字典序最小 若做不到返回空串
> 
> 找到前半部分比 a 大的改为 a 若不存在说明全是a 把最后一个改为+1（其实就是改为b）即可
> 
> > 需要稍加思考的是中间字符的处理 n为奇数则 5: 01-345 6: 012-345 中间字符不能动 所以 i < n/2

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    string breakPalindrome(string palindrome) {
        int n = palindrome.size();
        if (n <= 1) return "";
        for (int i = 0; i < n / 2; ++i) {
            if (palindrome[i] > 'a') {
                palindrome[i] = 'a';
                return palindrome;
            }
        }
        // if(palindrome.back() == 'z') return "";	// 不用
        // 因为可以从前到后遍历注定了后面不会比a大
        palindrome.back() = palindrome.back() + 1;
        return palindrome;
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

> [!NOTE] **[LeetCode 1352. 最后 K 个数的乘积](https://leetcode-cn.com/problems/product-of-the-last-k-numbers/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 想到了把0处理为1 以及使用前缀积 没想到遇到0就把前缀积更新为1 导致溢出出错

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class ProductOfNumbers {
public:
    vector<int> num, get;  //, zero;
    int zero;
    ProductOfNumbers() {
        num.push_back(1);
        get.push_back(1);
    }

    void add(int m) {
        if (m) {
            num.push_back(m);
            get.push_back((get[get.size() - 1]) * m);
        } else {
            zero = num.size();
            // zero.push_back(num.size());
            num.push_back(1);
            // 因为出现为0的 前面都已经没有价值了 可以直接赋值1
            // get.push_back(get[get.size()-1]);
            get.push_back(1);
        }
    }

    int getProduct(int k) {
        int res = 1, n = num.size();
        // for (auto v : zero) if (v >= n - k) return 0;
        if (zero >= n - k) return 0;
        return get[n - 1] / get[n - k - 1];
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

> [!NOTE] **[LeetCode 1574. 删除最短的子数组使剩余数组有序]()** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 需要注意题意：**删除一个子数组 子数组指一段连续的子序列**
> 
> 考虑从后面找到最长不上升序列，随后 **双指针**
> 
> - 右指针在后缀里移动, 每次固定左指针, 右指针往后试探到符合条件的位置
> 
> - 然后中间就是要删除的长度, 取最小值就是答案

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int findLengthOfShortestSubarray(vector<int>& arr) {
        int n = arr.size();
        int R = n - 1;
        for (int i = n - 1; i >= 0; --i)
            if (i == n - 1 || arr[i] <= arr[i + 1])
                R = i;
            else
                break;
        int res = R;
        for (int i = 0, j = R; i < R; ++i) {
            if (!i || arr[i] >= arr[i - 1]) {
                while (j < n && arr[j] < arr[i]) ++j;
                res = min(res, j - i - 1);
            } else
                break;
        }
        return max(res, 0);
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

> [!NOTE] **[LeetCode 1585. 检查字符串是否可以通过排序子字符串得到另一个字符串](https://leetcode-cn.com/problems/check-if-string-is-transformable-with-substring-sort-operations/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 因为排序不能改变相对位置 也即： 原本 x1 < x2 且边后 x1 在 x2 后面 比如 `8xxx5` 改之后还是 `8 5`
> 
> 排序会改变一段区间内元素的相对位置
> 
> 对于 s[i] 到 i 位置位置的每一个数 前面比它小的数的数量都必须小于等于 t[i] 位置前面比它小的数的数量
> 
> > 题解区有别的做法，如 `操作前后逆序对树木不增加即可转换` ，感觉并不优雅。

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp

class Solution {
public:
    bool isTransformable(string s, string t) {
        int n = s.size();
        vector<int> cntt(10), cnts(10);
        vector<vector<int>> lest(10, vector<int>()), less(10, vector<int>());
        for (int i = 0; i < n; ++i) {
            int v = t[i] - '0';
            int les = 0;
            for (int i = 0; i < v; ++i) les += cntt[i];
            lest[v].push_back(les);
            ++cntt[v];
        }
        for (int i = 0; i < n; ++i) {
            int v = s[i] - '0';
            int les = 0;
            for (int i = 0; i < v; ++i) les += cnts[i];
            less[v].push_back(les);
            int sz = less[v].size();
            if (sz > lest[v].size() || les > lest[v][sz - 1]) return false;
            ++cnts[v];
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

> [!NOTE] **[LeetCode 1702. 修改后的最大二进制字符串](https://leetcode-cn.com/problems/maximum-binary-string-after-change/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 模拟必然超时
> 
> 考虑：
> 
> > 0 1 1 1 1 0 ===> 1 0 1 1 1 1 消除第一位和末尾0 产生第二位0
> 
> 也即：每将最前面的0向后移动一位需要消耗掉后方一个0

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    string maximumBinaryString(string binary) {
        int n = binary.size(), k = 0;
        // 找到第一个0
        while (k < n && binary[k] == '1') ++ k ;
        if (k == n) return binary;
        
        int cnt = 0;
        for (int i = k + 1; i < n; ++ i )
            if (binary[i] == '0')
                ++ cnt;
        string res = string(n, '1');
        res[k + cnt] = '0';
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

> [!NOTE] **[LeetCode 1719. 重构一棵树的方案数](https://leetcode-cn.com/problems/number-of-ways-to-reconstruct-a-tree/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 非常非常非常好的综合思维题 难度拉满
> 
> TODO 重复做
> 
> 删点

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 如果一个点只有一个儿子 则可以缩点
    
    int checkWays(vector<vector<int>>& pairs) {
        // bitset 方便在比较的时候压位以降低复杂度
        bitset<501> g[501];
        for (auto & p : pairs) {
            int a = p[0], b = p[1];
            g[a][b] = g[b][a] = 1;
        }
        for (int i = 1; i <= 500; ++ i )
            if (g[i].count())
                g[i][i] = 1;
        
        int res = 1;
        set<int> rms;   // 记录删掉的点
        for (int i = 1; i <= 500; ++ i ) {
            // 循环删点
            if (!g[i].count()) continue;
            for (int j = i + 1; j <= 500; ++ j )
                if (g[i] == g[j]) {
                    res = 2;
                    rms.insert(j);
                    g[j][j] = 0;
                }
        }
        
        // 只保留单向关系
        vector<int> vers;
        for (auto & p : pairs) {
            int a = p[0], b = p[1];
            if (rms.count(a) || rms.count(b))
                g[a][b] = g[b][a] = 0;
            else vers.push_back(a), vers.push_back(b);
        }
        if (vers.size()) {
            sort(vers.begin(), vers.end());
            vers.erase(unique(vers.begin(), vers.end()), vers.end());
        }
        sort(vers.begin(), vers.end(), [&](int a, int b) {
            return g[a].count() > g[b].count();
        });
        
        for (auto & p : pairs) {
            int & a = p[0], & b = p[1];
            if (!rms.count(a) && !rms.count(b)) {
                if (g[a].count() == g[b].count()) return 0;
                else if (g[a].count() > g[b].count()) swap(a, b);
            }
        }
        
        for (auto & p : pairs) {
            int & a = p[0], & b = p[1];
            if (!rms.count(a) && !rms.count(b))
                g[a][b] = 0;
        }
        
        // 是否有一个点可以到达所有点
        if (vers.size() && g[vers[0]].count() != vers.size()) return 0;
        
        for (int i = 0; i < vers.size(); ++ i ) {
            int a = vers[i];
            if (!g[a].count()) continue;
            for (int j = i + 1; j < vers.size(); ++ j ) {
                int b = vers[j];
                // 此时ab必然是一条边
                if (g[a][b]) {
                    if ((g[a] & g[b]) != g[b]) return 0;
                    g[a] &= ~g[b];  // 删掉b子集
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

> [!NOTE] **[LeetCode 1733. 需要教语言的最少人数](https://leetcode-cn.com/problems/minimum-number-of-people-to-teach/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 要教的语言必定是最终 persons 中交流所使用的语言
> 
> 选择剩余 persons 中会的人最多的语言即可

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int minimumTeachings(int n, vector<vector<int>>& lg, vector<vector<int>>& fs) {
        int m = lg.size();
        vector<vector<bool>> g(m + 1, vector<bool>(n + 1));
        
        for (int i = 0; i < lg.size(); ++ i )
            for (auto x : lg[i])
                g[i + 1][x] = true;
        
        set<int> persons;
        for (auto & f : fs) {
            int x = f[0], y = f[1];
            bool flag = false;
            for (int i = 1; i <= n; ++ i )
                if (g[x][i] && g[y][i]) {
                    flag = true;
                    break;
                }
            if (flag) continue;
            persons.insert(x), persons.insert(y);
        }
        
        int s = 0;
        vector<int> cnt(n + 1);
        for (auto x : persons)
            for (int i = 1; i <= n; ++ i )
                if (g[x][i]) {
                    cnt[i] ++ ;
                    s = max(s, cnt[i]);
                }
        return persons.size() - s;
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

> [!NOTE] **[LeetCode 1753. 移除石子的最大得分](https://leetcode-cn.com/problems/maximum-score-from-removing-stones/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 如果不是 `一个堆大于剩下俩堆` 则必然可以取完

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    bool check(int a, int b, int c) {
        int cnt = 0;
        if (a) ++ cnt;
        if (b) ++ cnt;
        if (c) ++ cnt;
        return cnt > 1;
    }
    int maximumScore(int a, int b, int c) {
        int res = 0;
        while (check(a, b, c)) {
            int na = max(max(a, b), c);
            int nc = min(min(a, b), c);
            int nb = a + b + c - na - nc;
            a = na, b = nb, c = nc;
            int cnt = min(a, b) - c;
            if (cnt) {
                a -= cnt, b -= cnt;
                res += cnt;
            } else {
                // b == c
                if (a < 2) {
                    a -= 1, b -= 1;
                    res += 1;
                } else if (a / 2 >= b) {
                    cnt = b;
                    a -= cnt * 2, b -= cnt, c -= cnt;
                    res += cnt * 2;
                } else {
                    cnt = a / 2;
                    a -= cnt * 2, b -= cnt, c -= cnt;
                    res += cnt * 2;
                }
            }
            // cout << a << ' ' << b << ' ' << c << endl;
        }
        return res;
    }
};
```

##### **C++ 极简**

```cpp
class Solution {
public:
    int maximumScore(int a, int b, int c) {
        int d[] = {a, b, c};
        sort(d, d + 3);
        int x = 0;
        if (d[0] + d[1] < d[2]) x = d[2] - (d[0] + d[1]);
        else x = (a + b + c) % 2;
        return (a + b + c - x) / 2;
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

> [!NOTE] **[LeetCode 1775. 通过最少操作次数使数组的和相等](https://leetcode-cn.com/problems/equal-sum-arrays-with-minimum-number-of-operations/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 技巧性

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 每次挑选能够变化尽可能大的数值
    int minOperations(vector<int>& nums1, vector<int>& nums2) {
        int n1 = nums1.size(), n2 = nums2.size();
        int s1 = 0, s2 = 0;
        for (auto x : nums1) s1 += x;
        for (auto x : nums2) s2 += x;
        if (s1 < s2)
            swap(s1, s2), swap(n1, n2), swap(nums1, nums2);
        
        // s1 > s2
        // 可以达到的变化量(减少量)
        vector<int> cs(6, 0);
        for (auto x : nums1) cs[x - 1] ++ ;
        for (auto x : nums2) cs[6 - x] ++ ;
        
        int cur = 0, res = 0;
        for (int i = 5; i >= 1; -- i )
            if (cur + i * cs[i] < s1 - s2)
                cur += i * cs[i], res += cs[i];
            else {
                int dif = s1 - s2 - cur;
                int use = (dif + i - 1) / i;
                cur += i * use, res += use;
                break;
            }
        if (cur < s1 - s2) return -1;
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

> [!NOTE] **[LeetCode 1802. 有界数组中指定下标处的最大值](https://leetcode-cn.com/problems/maximum-value-at-a-given-index-in-a-bounded-array/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 注意细节 有好几个超时的 case
> 
> 另有二分做法

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int maxValue(int n, int index, int maxSum) {
        maxSum -= n;
        int v = 1, w = 0;
        while (maxSum) {
            int tot = min(index + w, n - 1) - max(index - w, 0) + 1;
            if (tot >= n) {
                // 扩张已经无用了
                v += maxSum / n;
                break;
            }
            if (maxSum >= tot) {
                maxSum -= tot;
                v ++ ;
                w ++ ;
            } else
                break;
        }
        return v;
    }
};
```

##### **C++ 二分**

大量用二分的 也可以 复杂度稍高一些

注意 用二分的方法在具体计算时仍要借助公式优化计算过程 否则仍然TLE

```cpp
class Solution {
public:
    long long getSum(int len, int mx) {
        long long a, b, more = 0;
        if (len >= mx) {
            a = 1, b = mx;
            more = len - mx;
        } else {
            a = mx - len + 1;
            b = mx;
            more = 0;
        }
        
        long long c = (b - a + 1);
        return (a + b) * c / 2 + more;
    }
    
    int maxValue(int n, int index, int maxSum) {
        int l = 0, r = maxSum;
        while (l < r) {
            int mid = (l + r + 1) / 2;
            long long sum = getSum(index + 1, mid) + getSum(n - index, mid) - mid;
            if (sum <= maxSum) {
                l = mid;
            } else {
                r = mid - 1;
            }
        }
        return l;
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

> [!NOTE] **[LeetCode LCP 33. 蓄水](https://leetcode-cn.com/problems/o8SXZn/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 思维题
> 
> 一开始在贪心，实际上可以考虑枚举的思路
> 
> 注意需特判 vat 的和为 0 的情况

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:    
    int storeWater(vector<int>& bucket, vector<int>& vat) {
        {
            int s = 0;
            for (auto v : vat)
                s += v;
            if (!s)
                return 0;
        }
        
        int n = bucket.size();
        unordered_map<int, int> hash;
        
        for (int i = 0; i < n; ++ i ) {
            int b = bucket[i], v = vat[i];
            for (int j = 1; j <= 1e4; ++ j )
                hash[j] += max((v + j - 1) / j - b, 0);
        }
        
        int res = INT_MAX;
        for (auto [k, v] : hash)
            res = min(res, k + v);
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

> [!NOTE] **[LeetCode 1878. 矩阵中最大的三个菱形和](https://leetcode-cn.com/problems/get-biggest-three-rhombus-sums-in-a-grid/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 暴力可过
> 
> 更优雅的实现 即写的时候想的利用前缀和思想
> 
> > 存以每个点为底点的两个斜方向的前缀和
> > 
> > 形如
> > 
> > \ / \ / \ / V (x, y) s1 计算左上斜着的和 s2 计算右上

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 暴力**

```cpp
class Solution {
public:
    vector<vector<int>> g;
    vector<int> t;
    int n, m;
    
    bool check(int x, int y, int d) {
        return x - d >= 0 && x + d < n && y - d >= 0 && y + d < m;
    }
    
    int sum(int x, int y, int d) {
        int ret = 0;
        if (d)
            ret = g[x - d][y] + g[x + d][y] + g[x][y - d] + g[x][y + d];
        else
            ret = g[x][y];
        
        for (int i = x - 1, j = y - d + 1; j < y; i -- , j ++ )
            ret += g[i][j];
        for (int i = x + 1, j = y - d + 1; j < y; i ++ , j ++ )
            ret += g[i][j];
        for (int i = x - 1, j = y + d - 1; j > y; i -- , j -- )
            ret += g[i][j];
        for (int i = x + 1, j = y + d - 1; j > y; i ++ , j -- )
            ret += g[i][j];
        return ret;
    }
    
    void get(int x, int y) {
        int d = 0;
        while (check(x, y, d)) {
            t.push_back(sum(x, y, d));
            d ++ ;
        }
    }
    
    vector<int> getBiggestThree(vector<vector<int>>& grid) {
        this->g = grid;
        n = g.size(), m = g[0].size();
        
        t.clear();
        for (int i = 0; i < n; ++ i )
            for (int j = 0; j < m; ++ j )
                get(i, j);
        sort(t.begin(), t.end());
        t.erase(unique(t.begin(), t.end()), t.end());
        reverse(t.begin(), t.end());
        
        vector<int> res;
        for (int i = 0; i < 3 && i < t.size(); ++ i )
            res.push_back(t[i]);
        return res;
    }
};
```

##### **C++**

```cpp
// yxc
const int N = 110;

int s1[N][N], s2[N][N];

class Solution {
public:
    vector<int> getBiggestThree(vector<vector<int>>& g) {
        memset(s1, 0, sizeof s1);
        memset(s2, 0, sizeof s2);
        int n = g.size(), m = g[0].size();
        for (int i = 1; i <= n; i ++ )
            for (int j = 1; j <= m; j ++ ) {
                s1[i][j] = s1[i - 1][j - 1] + g[i - 1][j - 1];
                s2[i][j] = s2[i - 1][j + 1] + g[i - 1][j - 1];
            }
        
        set<int> S;
        for (int i = 1; i <= n; i ++ )
            for (int j = 1; j <= m; j ++ ) {
                S.insert(g[i - 1][j - 1]);
                for (int k = 1; i - k >= 1 && i + k <= n && j - k >= 1 && j + k <= m; k ++ ) {
                    int a = s2[i][j - k] - s2[i - k][j];
                    int b = s1[i - 1][j + k - 1] - s1[i - k - 1][j - 1];
                    int c = s2[i + k - 1][j + 1] - s2[i - 1][j + k + 1];
                    int d = s1[i + k][j] - s1[i][j - k];
                    S.insert(a + b + c + d);
                }
                while (S.size() > 3) S.erase(S.begin());
            }
        return vector<int>(S.rbegin(), S.rend());
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

> [!NOTE] **[LeetCode 1888. 使二进制字符串字符交替的最少反转次数](https://leetcode-cn.com/problems/minimum-number-of-flips-to-make-the-binary-string-alternating/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
>
> **经典通用思路**
>
> 分析，经过若干轮转：
>
> 1. n 为偶数
>
>    总计只有 `010101...01` 和 `101010...10` 两种情况，取 min 即可
>
> 2. 两大类
>
>    2.1 `101010...01` 或 `010101...10` 首尾相同
>
>    2.2 `010101...11...01` 或 `101010...00...10` 中间有两个相同相同，两侧分别交替
>
> > 通用思路：
> >
> > **前后缀分解**
> >
> > > ```
> > > l[0][i];		// 以 0 从【起点】开始交替出现直到 i 的最小修改次数
> > > l[1][i];		// 以 1 从【起点】开始交替出现直到 i 的最小修改次数
> > > r[0][i];		// 以 0 从【末尾】开始交替出现直到 i 的最小修改次数
> > > r[1][i];		// 以 1 从【末尾】开始交替出现直到 i 的最小修改次数
> > > ```
> > >
> > > 则每个位置 `i` 和 `i + 1` 都可以 `O(1)` 求出。
> 
> 还有滑动窗口：最终一定是形成 010101 或 101010 这样的字符串，故直接**将当前串与目标串比较并维护窗口**的思路

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 前后缀分解**

```cpp
class Solution {
public:
    int minFlips(string s) {
        int n = s.size();
        vector<int> l[2], r[2];
        l[0] = l[1] = r[0] = r[1] = vector<int>(n);
        
        for (int i = 0; i < 2; ++ i )
            // c 当前变了多少个字母
            for (int j = 0, c = 0, k = i; j < n; ++ j , k ^= 1) {
                if (k != s[j] - '0')
                    c ++ ;
                l[i][j] = c;
            }
        for (int i = 0; i < 2; ++ i )
            for (int j = n - 1, c = 0, k = i; j >= 0; -- j , k ^= 1) {
                if (k != s[j] - '0')
                    c ++ ;
                r[i][j] = c;
            }
        
        if (n % 2 == 0)
            return min(l[0][n - 1], l[1][n - 1]);
        else {
            int res = min(l[0][n - 1], l[1][n - 1]);
            for (int i = 0; i + 1 < n; ++ i ) {
                res = min(res, l[0][i] + r[1][i + 1]);
                res = min(res, l[1][i] + r[0][i + 1]);
            }
            return res;
        }
    }
};
```

##### **C++ 滑动窗口1**

```cpp
class Solution {
public:
    int minFlips(string s) {
        int n = s.size(), cnt = 0;
        // 将字符串变成 01 串需要反转的次数
        string tar = "01";
        for (int i = 0; i < n; ++ i )
            cnt += (s[i] != tar[i % 2]);
        
        int res = min(cnt, n - cnt);
        s += s;
        for (int i = 0; i < n; ++ i ) {
            cnt -= (s[i] != tar[i % 2]);
            cnt += (s[i + n] != tar[(i + n) % 2]);
            res = min(res, min(cnt, n - cnt));
        }
        return res;
    }
};
```

##### **C++ 滑动窗口2**

```cpp
class Solution {
public:
    int minFlips(string s) {
        int n = s.size();
        s = s + s;
        
        string a, b;    // a = "0101...", b = "1010..."
        for (int i = 0; i < 2 * n; ++ i )
            a.push_back('0' + i % 2), b.push_back('0' + (i + 1) % 2);
        
        int res = n, da = 0, db = 0;
        for (int i = 0; i < 2 * n; ++ i ) {
            if (s[i] != a[i])
                da ++ ;
            if (s[i] != b[i])
                db ++ ;
            
            // 维护窗口的实现
            if (i >= n) {
                if (s[i - n] != a[i - n])
                    da -- ;
                if (s[i - n] != b[i - n])
                    db -- ;
            }
            if (i >= n - 1)
                res = min(res, min(da, db));
        }
        return res;
    }
};

// 转化为常见方式
class Solution {
public:
    int minFlips(string s) {
        int n = s.size();
        s = s + s;
        
        string a, b;    // a = "0101...", b = "1010..."
        for (int i = 0; i < 2 * n; ++ i )
            a.push_back('0' + i % 2), b.push_back('0' + (i + 1) % 2);
        
        int res = n, da = 0, db = 0;
        for (int i = 0; i < n; ++ i ) {
            if (s[i] != a[i])
                da ++ ;
            if (s[i] != b[i])
                db ++ ;
        }
        
        res = min(res, min(da, db));
        
        for (int i = n; i < 2 * n; ++ i ) {
            if (s[i] != a[i])
                da ++ ;
            if (s[i] != b[i])
                db ++ ;
            
            {
                if (s[i - n] != a[i - n])
                    da -- ;
                if (s[i - n] != b[i - n])
                    db -- ;
            }
            
            res = min(res, min(da, db));
        }
        return res;
    }
};
```

##### **C++ 自己思路TLE**

```cpp
// 51 / 65 个通过测试用例
class Solution {
public:
    const static int N = 2e5 + 10;
    int n;
    int f[N];
    string ns;
    
    int get(int l, int r) {
        int c = 0;
        for (int i = l, flag = 1; i < r; i += f[i], flag ^= 1 )
            if (flag == 0)
                c += min(r - i, f[i]);
        
        return min(c, n - c);
    }
    
    int minFlips(string s) {
        n = s.size();
        ns = s + s;
        
        memset(f, 0, sizeof f);
        f[2 * n] = 1;
        for (int i = 2 * n - 1; i > 0; -- i )
            if (ns[i - 1] != ns[i])
                f[i] = f[i + 1] + 1;
            else
                f[i] = 1;
        
        int res = n;
        for (int i = 1; i <= n; i += f[i] ) {
            int t = get(i, i + n);
            // cout << "i = " << i << " t = " << t << endl;
            res = min(res, min(t, n - t));
        }
        // cout << endl;
        
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

> [!NOTE] **[LeetCode 1930. 长度为 3 的不同回文子序列](https://leetcode-cn.com/problems/unique-length-3-palindromic-subsequences/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 注意 l & r 避免数值溢出
> 
> 另有 trick 的思想和解法: set 可以用数组标记替代

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int countPalindromicSubsequence(string s) {
        int n = s.size();
        vector<vector<int>> sum(n + 1, vector<int>(26));
        
        for (int i = 1; i <= n; ++ i ) {
            int v = s[i - 1] - 'a';
            for (int j = 0; j < 26; ++ j )
                if (j == v)
                    sum[i][j] = sum[i - 1][j] + 1;
                else
                    sum[i][j] = sum[i - 1][j];
        }
        
        unordered_set<string> S;
        for (int i = 2; i < n; ++ i ) {
            int v = s[i - 1] - 'a';
            for (int j = 0; j < 26; ++ j ) {
                int l = sum[i - 1][j], r = sum[n][j] - sum[i][j];
                if (l && r) {   // ATTENTION && 而不是 *
                    string t;
                    t.push_back('a' + j);
                    t.push_back('a' + v);
                    t.push_back('a' + j);
                    S.insert(t);
                }
            }
        }
        return S.size();
    }
};
```

##### **C++ trick**

```cpp
class Solution {
public:
    int countPalindromicSubsequence(string s) {
        int n = s.size();
        vector<int> u(26), v(26);
        for (int i = 0; i < n; ++i)
            u[s[i]-'a'] ++;
        vector<vector<int>> f(26, vector<int>(26));
        for (int i = 0; i < n; ++i) {
            u[s[i]-'a'] --;
            for (int c = 0; c < 26; ++c)
                if (u[c] && v[c]) f[s[i]-'a'][c] = 1;
            v[s[i]-'a'] ++;
        }
        int res = 0;
        for (int a = 0; a < 26; ++a)
            for (int b = 0; b < 26; ++b)
                res += f[a][b];
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

> [!NOTE] **[LeetCode 1975. 最大方阵和](https://leetcode-cn.com/problems/maximum-matrix-sum/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 简单推理易知，总是能【通过传递的方式】同时使两个位置的值取反
> 
> 显然扫一遍统计即可

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// 虚拟比赛代码
class Solution {
public:
    using LL = long long;
    
    int n;
    
    long long maxMatrixSum(vector<vector<int>>& matrix) {
        this->n = matrix.size();
        LL s = 0, ns = 0, mv = INT_MAX, msv = INT_MIN, c = 0;
        for (int i = 0; i < n; ++ i )
            for (int j = 0; j < n; ++ j ) {
                LL t = matrix[i][j];
                if (t >= 0)
                    s += t, mv = min(mv, t);
                else
                    ns += t, msv = max(msv, t), c ++ ;
            }
        if (c & 1) {
            if (-msv > mv)
                s = s - ns - mv - mv;
            else
                s = s - (ns - msv) + msv;
        } else
            s -= ns;
        return s;
    }
};
```

##### **C++ 简化**

```cpp
// 进一步简化 by Heltion
class Solution {
public:
    long long maxMatrixSum(vector<vector<int>>& matrix) {
        int n = 0, mn = 100000;
        long long sum = 0;
        for (auto& v : matrix) for (int x : v) {
            sum += abs(x);
            if (x < 0) n += 1;
            mn = min(abs(x), mn);
        }
        if (n & 1) return sum - mn * 2;
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

> [!NOTE] **[LeetCode 1982. 从子集的和还原数组](https://leetcode-cn.com/problems/find-array-given-subset-sums/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 核心在于将负数集合转化为正数集合
> 
> 随后简化处理流程
> 
> > https://leetcode-cn.com/problems/find-array-given-subset-sums/solution/ti-jie-cong-zi-ji-de-he-huan-yuan-shu-zu-q9qw/
> > 
> > 全非负数的基础版 题目链接 https://www.codechef.com/problems/ANUMLA
> > 
> > 理解思路
> 
> > https://leetcode-cn.com/problems/find-array-given-subset-sums/solution/jian-yi-ti-jie-by-sfiction-9i43/
> > 
> > 本题有负值，则将所有数 -min_element 转化为非负问题
> > 
> > [非负问题中最小值必为0 次小值就是原集合的最小值] ===> [最大值与最小值差即为该数的绝对值 正负性待定]。递归求解

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    vector<int> recoverArray(int n, vector<int>& a) {
        sort(a.begin(), a.end());
        int B = -a[0];
        
        // 非负集合
        vector<int> b;
        for (auto x : a)
            b.push_back(B + x);
        
        vector<int> res;
        for (int _ = 0; _ < n; ++ _ ) {
            // 最小值为空集 次小值-最小值即为当前最小的数的绝对值
            int t = b[1] - b[0];
            res.push_back(t);
            
            // d 替代set实现值的删除
            int m = b.back();
            vector<int> d(m + 1);
            for (auto x : b)
                d[x] ++ ;
            b.clear();
            // ATTENTION
            for (int i = 0; i <= m; ++ i )
                if (d[i]) {
                    b.push_back(i);
                    d[i] -- ;
                    d[i + t] -- ;
                    i -- ; // 回退 继续检查当前值
                }
        }
        
        // 枚举子集
        for (int i = 0; i < 1 << n; ++ i ) {
            int t = 0;
            for (int j = 0; j < n; ++ j )
                if (i >> j & 1)
                    t += res[j];
            if (t == B) {
                // 则该子集中的所有数字都应是负数
                for (int j = 0; j < n; ++ j )
                    if (i >> j & 1)
                        res[j] *= -1;
                return res;
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

> [!NOTE] **[LeetCode 1996. 游戏中弱角色的数量](https://leetcode-cn.com/problems/the-number-of-weak-characters-in-the-game/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
>
> **重点在于：单纯的按照两个维度 升/降 序排列会遇到无法区分的问题**
>
> **为了达到区分效果 两个维度分别降升序**
>
> > Trick 思维题 利用排序规则
>
> 显然可以先按某一维度排序 如 `按第一维度从大到小排序`
>
> 排序后，考虑遍历：
>
> - 如果第二维度以同样规则 `从大到小` 排序，则无法严格区分在 **第一维度上 `大于/等于`** 当前遍历位置的元素
>
> - 考虑第二维度按相反规则 `从小到大` 排序，则可以在遍历时维护第二维度的最大值
>
>   此时：
>
>   - 该最大值一定代表着 **第一维度上 `大于`** 当前遍历位置元素的第二维度值【可以区分 `大于/等于`】
>   - 第一维度相等的元素，内部不会对结果造成影响，因为第二维度较小的总是在前面

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    vector<vector<int>> ps;
    
    int numberOfWeakCharacters(vector<vector<int>>& properties) {
        this->ps = properties;
        // ATTENTION [排序trick] 和 [维护规则]
        sort(ps.begin(), ps.end(), [](const vector<int> & a, const vector<int> & b) {
            if (a[0] == b[0])
                return a[1] < b[1];
            return a[0] > b[0];
        });
        int n = ps.size(), maxd = 0, res = 0;
        for (auto & p : ps) {
            if (p[1] < maxd)    // 排序规则注定此时 p[0] 必定不同
                res ++ ;
            else
                maxd = p[1];
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

> [!NOTE] **[LeetCode 2009. 使数组连续的最少操作数](https://leetcode-cn.com/problems/minimum-number-of-operations-to-make-array-continuous/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 一开始在思考将数据分段再贪心
> 
> 其实是个思维题
> 
> 要想到【**最终结果一定是一个长度为 n 的连续序列**】
> 
> 另外 数字范围极大所以不能遍历数值 而是遍历输入数据同时使用【滑动窗口】维护 也因此需要【对输入进行排序

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 思维 trick
    // 想象一下所有nums在一个数轴上，然后你拿一个长度为n的窗口去扫
    // 那么对于每个特定的窗口，里面有几个点就有几个不需要挪的坑。

    // 数值范围较大 1e9 所以无法直接使用 st[] 或者 set 记录然后直接遍历值
    // 应该 排序(must) + 滑动窗口
    
    int minOperations(vector<int>& nums) {
        int n = nums.size();
        sort(nums.begin(), nums.end());
        
        int res = 2e9;
        // 还需要处理 nums 中重复的数字
        unordered_map<int, int> hash;
        for (int l = 0, r = 0, cnt = 0; r < n; ++ r ) {
            while (l <= r && nums[l] <= nums[r] - n) {
                if ( -- hash[nums[l]] == 0)
                    cnt -- ;
                l ++ ;
            }
            if ( ++ hash[nums[r]] == 1)
                cnt ++ ;
            res = min(res, n - cnt);
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

> [!NOTE] **[LeetCode 2025. 分割数组的最多方案数](https://leetcode-cn.com/problems/maximum-number-of-ways-to-partition-an-array/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 显然有分情况讨论
> 
> **重点在于将修改一个数造成后缀部分全部被动修改的结果化为偏移**
> 
> **维护偏移量来统一计数**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using LL = long long;
    
    int waysToPartition(vector<int>& nums, int k) {
        int n = nums.size(), res = 0;
        
        vector<LL> s(n);
        for (int i = 0; i < n; ++ i )
            s[i] = (i ? s[i - 1] : 0) + nums[i];
        
        {
            // do not modify
            int ret = 0;
            for (int i = 0; i < n - 1; ++ i )
                if (s[i] == s[n - 1] - s[i])
                    ret ++ ;
            res = max(res, ret);
        }
        {
            // modify once
            // ATTENTION 枚举每个位置 通过hash-table找出符合要求的分界点的数量
            // 改变位置分别是 [前缀or后缀] 时的出现情况
            unordered_map<LL, int> l, r;
            // 细节 n - 1
            for (int i = 0; i < n - 1; ++ i )
                l[s[i]] ++ ;
            for (int i = n - 1; i >= 0; -- i ) {
                // 细节 n - 1
                if (i < n - 1)
                    l[s[i]] -- , r[s[i]] ++ ;
                
                // change the i-th item, get new sum
                LL sum = s[n - 1] - nums[i] + k;
                if (sum & 1)
                    continue;
                LL tar = sum >> 1;
                // ATTENTION: tar - k + nums[i]
                // 核心在于改变当前数字后后缀都会发生偏移 所以找偏移前的数值
                res = max(res, l[tar] + r[tar - k + nums[i]]);
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

### 动态统计计数题

> [!NOTE] **[LeetCode 1224. 最大相等频率](https://leetcode-cn.com/problems/maximum-equal-frequency/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
>
> `unordered_map + map` 计数动态统计题
>
> 类似的还有 Weekly 12 ：[1244. 力扣排行榜](https://leetcode-cn.com/problems/design-a-leaderboard/)

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    map<int, int> s;
    map<int, int>::iterator it;
    void del(int x) {
        if (s[x] > 1)
            --s[x];
        else
            s.erase(x);
    }
    void ins(int x) { ++s[x]; }
    bool check() {
        if (s.size() > 2) return false;
        // 只有一个字符 或一个字符出现连续多次
        int c0 = s.begin()->first, n0 = s.begin()->second;
        if (s.size() == 1) return c0 == 1 || n0 == 1;

        // 此时 size = 2
        it = s.begin();
        ++it;
        int c1 = it->first, n1 = it->second;
        // 有两个不同的出现次数 则第一个次数为0且只出现一次
        //  或 第一个出现次数比第二个小1且只出现1次
        return c0 == 1 && n0 == 1 || c1 == c0 + 1 && n1 == 1;
    }
    int maxEqualFreq(vector<int>& nums) {
        // unordered_map key->cnt
        // map cnt->num
        int n = nums.size(), v, res = 0;
        unordered_map<int, int> kc;
        for (int i = 0; i < n; ++i) {
            v = nums[i];
            if (kc[v]) del(kc[v]);
            ins(++kc[v]);
            if (check()) res = i;
        }
        return res + 1;
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

> [!NOTE] **[LeetCode 343. 整数拆分](https://leetcode-cn.com/problems/integer-break/)**
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
    int integerBreak(int n) {
        if (n < 3) return 1;
        else if (n == 3) return 2;
        int res = 1;
        if (n % 3 == 1) n -= 4, res *= 4;
        while (n >= 3) n -= 3, res *= 3;
        if (n) res *= n;
        return res;
    }
};
```

##### **C++**

```cpp
class Solution {
public:
    int integerBreak(int n) {
        if (n <= 3) return 1 * (n - 1);
        int p = 1;
        while (n >= 5) n -= 3, p *= 3;
        return p * n;
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

### 跳跃游戏

> [!NOTE] **[LeetCode 55. 跳跃游戏](https://leetcode-cn.com/problems/jump-game/)**
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
    bool canJump(vector<int>& nums) {
        int n = nums.size();
        int l = n-1;
        for (int i = n - 1; i >= 0; -- i ) {
            if (i + nums[i] >= l) l = i;
        }
        return l == 0;
    }
};

class Solution {
public:
    bool canJump(vector<int>& nums) {
        int n = nums.size(), k = 0;
        for (int i = 0; i < n; ++ i ) {
            if (k < i)
                return false;
            k = max(k, i + nums[i]);
        }
        return true;
    }
};
```

##### **Python**

```python
"""
本贪心算法的核心：如果一个位置能够到达，那么这个位置左侧所有位置都能到达
1. 用一个变量记录，前一个位置能到达最远的位置
2. 若前i - 1个位置中跳，跳到最远的位置是 max_i 比 i 小，表示从前i - 1个位置中跳，跳不到i的位置，因此一定不能跳到最后一个的位置
3. 若前i - 1个位置中跳，能跳到i，则继续尝试从i位置跳，可能会跳得更远，更新 max_i 的值
"""

class Solution:
    def canJump(self, nums):
        #初始化当前能到达最远的位置
        last = 0      
        #i为当前位置，jump是当前位置的跳数
        for i, jump in enumerate(nums):   
            # 更新最远能到达位置
            if last >= i and i + jump > last:  
                last = i + jump  
        return last >= i 
  
  
# 写法2  
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        last = 0 
        for i, jump in enumerate(nums):
            if last < i:
                return False
            last = max(last, i + jump)
        return True
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 45. 跳跃游戏 II](https://leetcode-cn.com/problems/jump-game-ii/)**
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
    int jump(vector<int>& nums) {
        int n = nums.size();
        int res = 0, ed = 0, mx = 0;
        for (int i = 0; i < n - 1; ++ i ) {
            mx = max(mx, i + nums[i]);
            if (i == ed) {
                res ++ ;
                ed = mx;
            }
        }
        return res;
    }
};
```

##### **Python**

```python
# 普通dp：会超时
"""
【有限集合的最优化问题】，用dp来做：
1. 状态定义 f[i]: 表示跳到点 i 需要的最小步数
2. 状态转移 从第0个点、1个点... j个点跳到 第i个点的步数最小值（前提是：能跳到）
3. 能跳到: j + nums[j] >= i
"""
class Solution:
    def jump(self, nums: List[int]) -> int:
        n = len(nums)
        f = [float('inf')] * n
        f[0] = 0
        for i in range(n):
            for j in range(i):
                if j + nums[j] >= i:
                    f[i] = min(f[i], f[j] + 1)
        return f[n-1]
      

# dp + 贪心优化
"""
f[i] 其实具有单调性，也就是f[i+1] >= f[i], 可以用反证法：
1. 如果f[i+1] < f[i], 设k(k<=i)点 跳到 i+1，那么就有k+nums[k] >= i+1,那k+nums[k] 必定也大于i, 此时 f[i+1]=f[i]；
2. 如果nums的每一项都为1，那么f[i+1] > f[i]，综上 与假设矛盾，所以 f[i]具有单调性
3. 有了单调性后，可以用第一个能跳到i的点更新了，因为第一个能跳到，那后面的肯定也可以，但是后面的f[j+1] >= f[j], 所以用第一个就可以了，也就是尽可能的选择靠前的点，这样步数可能会减少，有点贪心的思想。
4. j 就从0 开始，当第一次 j + nums[j] >= f[i]的时候，用f[j]来更新f[i], 也就是 f[i] = f[j] + 1
"""

class Solution:
    def jump(self, nums: List[int]) -> int:
        n = len(nums)
        f = [float('inf')] * n
        f[0] = 0
        j = 0
        for i in range(1, n):
            while j + nums[j] < i:
                j += 1
            f[i] = f[j] + 1
        return f[n-1]


# 借助变量来求解
"""

"""
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 1306. 跳跃游戏 III](https://leetcode-cn.com/problems/jump-game-iii/)**
> 
> 题意: 每个位置上可以左右跳 $arr[i]$ 个距离

> [!TIP] **思路**
> 
> 裸搜索

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int n;
    bool dfs(vector<int>& arr, vector<bool>& vis, int s) {
        if (arr[s] == 0) return true;
        vis[s] = true;
        bool l = false, r = false;
        if (s - arr[s] >= 0 && !vis[s - arr[s]]) l = dfs(arr, vis, s - arr[s]);
        if (s + arr[s] < n && !vis[s + arr[s]]) r = dfs(arr, vis, s + arr[s]);
        return l || r;
    }
    bool canReach(vector<int>& arr, int start) {
        this->n = arr.size();
        bool nofind = true;
        for (int i = 0; i < n; ++i)
            if (!arr[i]) {
                nofind = false;
                break;
            }
        if (nofind) return false;
        vector<bool> vis(n);
        return dfs(arr, vis, start);
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

> [!NOTE] **[LeetCode 1345. 跳跃游戏 IV](https://leetcode-cn.com/problems/jump-game-iv/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> bfs 需要注意的是使用map记录优化
> 
> 建图跑最短路也可以 n^2

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int minJumps(vector<int>& arr) {
        int n = arr.size();
        unordered_map<int, vector<int>> m;
        for (int i = 0; i < n; ++i) m[arr[i]].push_back(i);
        vector<int> dis(n, INT_MAX);
        vector<int> vis(n, false);
        dis[n - 1] = 0;
        vis[n - 1] = true;
        queue<int> q;
        q.push(n - 1);
        while (!q.empty()) {
            int u = q.front();
            q.pop();
            if (u && !vis[u - 1] && m.find(arr[u - 1]) != m.end()) {
                dis[u - 1] = min(dis[u - 1], dis[u] + 1);
                vis[u - 1] = true;
                q.push(u - 1);
            }
            if (u + 1 < n && !vis[u + 1] && m.find(arr[u + 1]) != m.end()) {
                dis[u + 1] = min(dis[u + 1], dis[u] + 1);
                vis[u + 1] = true;
                q.push(u + 1);
            }
            if (m.find(arr[u]) != m.end()) {
                for (auto v : m[arr[u]]) {
                    if (!vis[v]) {
                        vis[v] = true;
                        dis[v] = min(dis[v], dis[u] + 1);
                        q.push(v);
                    }
                }
                m.erase(arr[u]);
            }
        }
        return dis[0];
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

> [!NOTE] **[LeetCode 1340. 跳跃游戏 V](https://leetcode-cn.com/problems/jump-game-v/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> dp 记忆化搜索即可

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int dfs(vector<int>& arr, vector<int>& dp, int n, int d, int x) {
        if (dp[x]) return dp[x];
        dp[x] = 1;
        for (int i = x - 1; i >= max(0, x - d); --i) {
            // check
            if (arr[i] >= arr[x]) break;
            dp[x] = max(dp[x], dfs(arr, dp, n, d, i) + 1);
        }
        for (int i = x + 1; i <= min(n - 1, x + d); ++i) {
            // check
            if (arr[i] >= arr[x]) break;
            dp[x] = max(dp[x], dfs(arr, dp, n, d, i) + 1);
        }
        return dp[x];
    }
    int maxJumps(vector<int>& arr, int d) {
        int n = arr.size(), res = 0;
        vector<int> dp(n);
        for (int i = 0; i < n; ++i) { res = max(res, dfs(arr, dp, n, d, i)); }
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

> [!NOTE] **[LeetCode 1871. 跳跃游戏 VII](https://leetcode-cn.com/problems/jump-game-vii/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> - BIT维护
> 
>   当前 x 位置能否跳到取决于 [x - maxJump, x - minJump] 区间是否有可达点
> 
> - DP
> 
>   判断是否有合法方案，维护到某个位置位置可以跳到的所有位置的个数(前缀和)

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ BIT**

```cpp
class Solution {
public:
    const static int N = 2e5 + 10;
    int tr[N], n;
    
    int lowbit(int x) {
        return x & -x;
    }
    void add(int x, int c) {
        for (int i = x; i <= n; i += lowbit(i))
            tr[i] += c;
    }
    int sum(int x) {
        int res = 0;
        for (int i = x; i; i -= lowbit(i))
            res += tr[i];
        return res;
    }
    
    bool canReach(string s, int minJump, int maxJump) {
        memset(tr, 0, sizeof tr);
        this->n = s.size();
        add(1, 1);
        for (int i = 2; i <= n; ++ i )
            if (s[i - 1] == '0') {
                int l = max(i - maxJump - 1, 0), r = max(i - minJump, 0);
                if (sum(r) - sum(l) > 0) {
                    add(i, 1);
                }
            }
        return sum(n) - sum(n - 1) > 0;
    }
};
```

##### **C++ DP**

```cpp
class Solution {
public:
    bool canReach(string str, int a, int b) {
        int n = str.size();
        vector<int> f(n + 1), s(n + 1);
        f[1] = 1;
        s[1] = 1;
        for (int i = 2; i <= n; i ++ ) {
            if (str[i - 1] == '0' && i - a >= 1) {
                int l = max(1, i - b), r = i - a;
                if (s[r] > s[l - 1]) f[i] = 1;
            }
            s[i] = s[i - 1] + f[i];
        }
        return f[n];
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

> [!NOTE] **[LeetCode 763. 划分字母区间](https://leetcode-cn.com/problems/partition-labels/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 经典边界思路

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    vector<int> partitionLabels(string S) {
        unordered_map<int, int> last;
        for (int i = 0; i < S.size(); i ++ ) last[S[i]] = i;
        vector<int> res;
        int start = 0, end = 0;
        for (int i = 0; i < S.size(); i ++ ) {
            end = max(end, last[S[i]]);
            if (i == end) {
                res.push_back(end - start + 1);
                start = end = i + 1;
            }
        }
        return res;
    }
};
```

##### **Python**

```python
# 模拟题：边扫描 边输出每段的长度即可。
# 用哈希表 记录每个字母最后出现的位置。从前往后扫描的时候，需要定义这一段的起始位置和终止位置。

class Solution:
    def partitionLabels(self, s: str) -> List[int]:
        my_dict = dict(); res = []
        for i in range(len(s)):
            my_dict[s[i]] = i 
        l, r = 0, 0 
        for i in range(len(s)):
            r = max(r, my_dict[s[i]])
            if r == i:
                res.append(r - l + 1)
                r = l = i + 1
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

### 转化为图论

> [!NOTE] **[LeetCode 765. 情侣牵手](https://leetcode-cn.com/problems/couples-holding-hands/)**
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
    // 经典图论模型 环状图 每次交换环中断
    // 目标 把环状图变为若干个自环

    vector<int> p;

    int find(int x) {
        if (p[x] != x)
            p[x] = find(p[x]);
        return p[x];
    }

    int minSwapsCouples(vector<int>& row) {
        int n = row.size() / 2;
        for (int i = 0; i < n; ++ i )
            p.push_back(i);
        
        int cnt = n;
        for (int i = 0; i < n * 2; i += 2 ) {
            int a = row[i] / 2, b = row[i + 1] / 2;
            // 环等价于联通块 找环即找联通块
            if (find(a) != find(b)) {
                p[find(a)] = find(b);
                cnt -- ;
            }
        }
        return n - cnt;
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