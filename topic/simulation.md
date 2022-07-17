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

> [!NOTE] **[Luogu 闰年判断](https://www.luogu.com.cn/problem/P5711)**
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

bool f(int x) {
    return x % 4 == 0 && x % 100 || x % 400 == 0;
}

int main() {
    cin >> n;
    
    cout << (f(n) ? 1 : 0) << endl;
    
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

> [!NOTE] **[LeetCode 1154. 一年中的第几天](https://leetcode-cn.com/problems/day-of-the-year/)**
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
    int days[13] = {0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    int dayOfYear(string date) {
        int year = stoi(date.substr(0, 4).c_str());
        int mon = stoi(date.substr(5, 2).c_str());
        int day = stoi(date.substr(8, 2).c_str());
        // cout << year << ' ' << mon << ' ' << day << endl;
        
        int res = 0;
        for (int i = 1; i < mon; ++ i ) {
            res += days[i];
            if (i == 2)
                if (year % 400 == 0 || year % 100 != 0 && year % 4 == 0)
                    ++ res;
        }
        res += day;
        return res;
    }
};
```

##### **C++ sscanf**

```cpp
int days[] = {0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
class Solution {
public:
    int ordinalOfDate(string date) {
        int y, m, d;
        sscanf(date.c_str(), "%d-%d-%d", &y, &m, &d);
        int ret = 0;
        for (int i = 1; i < m; ++ i) {
            int day = days[i];
            if (y % 400 == 0 || y % 4 == 0 && y % 100 != 0) if (i == 2) day ++;
            ret += day;
        }
        return ret + d;
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

> [!NOTE] **[LeetCode 1185. 一周中的第几天](https://leetcode-cn.com/problems/day-of-the-week/)**
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
    bool leap(int x) {
        return x % 4 == 0 && x % 100 != 0 && x || x % 400 == 0;
    }
    vector<int> days = {0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    vector<string> res = {"Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"};
    string dayOfTheWeek(int day, int month, int year) {
        int ret = 4;
        for (int i = 1971; i < year; ++ i )
            if (leap(i)) ret += 366;
            else ret += 365;
        for (int i = 1; i < month; ++ i )
            if (leap(year) && i == 2) ret += 29;
            else ret += days[i];
        ret += day;
        ret %= 7;
        return res[ret];
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

> [!NOTE] **[LeetCode 1360. 日期之间隔几天](https://leetcode-cn.com/problems/number-of-days-between-two-dates/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 计算两个日期的天数差

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 模拟**

```cpp
class Solution {
    bool leap_year(int year) {
         return ((year % 400 == 0) || (year % 100 != 0 && year % 4 == 0));
    }
    int date_to_int(string date) {
        int year, month, day;
        sscanf(date.c_str(), "%d-%d-%d", &year, &month, &day);
        int month_length[] = {31, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30};
        int ans = 0;
        while (year != 1971 or month != 1 or day != 1) {
            ++ans;
            if (--day == 0)
                if (--month == 0)
                    --year;
            if (day == 0) {
                day = month_length[month];
                if (month == 2 && leap_year(year))
                    ++day;
            }
            if (month == 0)
                month = 12;
        }
        return ans;
    }
public:
    int daysBetweenDates(string date1, string date2) {
        return abs(date_to_int(date1) - date_to_int(date2));
    }
};
```

##### **C++ 赛榜**

```cpp
class Solution {
    int x[100] = {0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    int get(string date) {
        int y = 0, m = 0, d = 0, i, ans = 0;
        for (i = 0; i < 4; i++) y = y * 10 + date[i] - '0';
        for (i = 5; i < 7; i++) m = m * 10 + date[i] - '0';
        for (i = 8; i < 10; i++) d = d * 10 + date[i] - '0';
        for (i = 0; i < y; i++)
            if (i % 400 == 0 || i % 4 == 0 && i % 100)
                ans += 366;
            else
                ans += 365;
        if (y % 400 == 0 || y % 4 == 0 && y % 100)
            x[2] = 29;
        else
            x[2] = 28;
        for (i = 1; i < m; i++) ans += x[i];
        ans += d;
        return ans;
    }

public:
    int daysBetweenDates(string date1, string date2) {
        return abs(get(date1) - get(date2));
    }
};
```

##### **C++ Zeller公式**

```cpp
class Solution {
public:
    int toDay(const string& dateStr) {
        int year, month, day;
        sscanf(dateStr.c_str(), "%d-%d-%d", &year, &month, &day);
        if (month <= 2) {
            year--;
            month += 10;
        } else
            month -= 2;
        return 365 * year + year / 4 - year / 100 + year / 400 + 30 * month +
               (3 * month - 1) / 5 + day /* -584418 */;
    }
    int daysBetweenDates(string date1, string date2) {
        return abs(toDay(date1) - toDay(date2));
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

> [!NOTE] **[Luogu USACO1.2 方块转换 Transformations](https://www.luogu.com.cn/problem/P1205)**
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
#include<bits/stdc++.h>
using namespace std;

using VS = vector<string>;

int n;

void mirror(VS & s) {
    for (int i = 0; i < n; ++ i )
        for (int j = 0, k = n - 1; j < k; ++ j , -- k )
            swap(s[i][j], s[i][k]);
}

void rotate(VS & s) {
    // 关于对角线对称
    for (int i = 0; i < n; ++ i )
        for (int j = 0; j < i; ++ j )
            swap(s[i][j], s[j][i]);
    // 镜像对称
    mirror(s);
}

int check(VS & a, VS & b) {
    auto c = a;
    for (int i = 1; i <= 3; ++ i ) {
        rotate(c);
        if (c == b) return i;
    }
    c = a;
    mirror(c);
    if (c == b) return 4;
    for (int i = 1; i <= 3; ++ i ) {
        rotate(c);
        if (c == b) return 5;
    }
    if (a == b) return 6;
    return 7;
}

int main() {
    VS a, b;
    string line;
    
    cin >> n;
    for (int i = 0; i < n; ++ i ) cin >> line, a.push_back(line);
    for (int i = 0; i < n; ++ i ) cin >> line, b.push_back(line);
    cout << check(a, b) << endl;
    
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

> [!NOTE] **[Luogu NOIP2003 普及组 乒乓球](https://www.luogu.com.cn/problem/P1042)**
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

string get(string s) {
    string ret;
    for (auto c : s)
        if (c != 'E')
            ret.push_back(c);
        else
            break;
    return ret;
}

void f(string s, int d) {
    int n = s.size();
    
    // 如果一局比赛刚开始 则此时比分为 0:0 最后一个case
    int w = 0, l = 0;
    
    int i = 0;
    while (i < n) {
        int j = i;
        // case
        while (j < n && (w < d && l < d || w >= d - 1 && l >= d - 1 && abs(w - l) < 2)) {
            if (s[j] == 'W')
                w ++ ;
            else
                l ++ ;
            j ++ ;
        }
        
        if ((w >= d || l >= d) && abs(w - l) >= 2) {
            cout << w << ':' << l << endl;
            w = 0, l = 0;
        }
        i = j;
    }
    cout << w << ':' << l << endl;
}

int main() {
    string s, str;
    while (cin >> str)
        s += str;
    
    s = get(s);
    
    f(s, 11);
    cout << endl;
    f(s, 21);
    
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

> [!NOTE] **[Luogu NOIP2006 提高组 作业调度方案](https://www.luogu.com.cn/problem/P1065)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 经典大模拟

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 22, M = 1e5 + 10;

int m, n;
int all[N * N];
int id[N][N], cost[N][N], step[N], pred[N];
bool busy[N][M]; // the machine is idle

int main() {
    cin >> m >> n;
    
    for (int i = 1; i <= m * n; ++ i )
        cin >> all[i];
    
    for (int i = 1; i <= n; ++ i )
        for (int j = 1; j <= m; ++ j )
            cin >> id[i][j];
    for (int i = 1; i <= n; ++ i )
        for (int j = 1; j <= m; ++ j )
            cin >> cost[i][j];
    
    int res = 0;
    for (int i = 1; i <= m * n; ++ i ) {
        int sth = all[i];
        step[sth] ++ ;
        
        int tid = id[sth][step[sth]], tcost = cost[sth][step[sth]];
        
        // begin from last done time
        for (int j = pred[sth] + 1; ; ++ j )
            if (!busy[tid][j]) {
                int k = j;
                while (k - j < tcost && !busy[tid][k])
                    k ++ ;
                if (k - j == tcost) {
                    for (int t = j; t < k; ++ t )
                        busy[tid][t] = true;
                    pred[sth] = k - 1;
                    // update
                    res = max(res, k - 1);
                    break;
                }
                // j = k - 1;
                j = k;
            }
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



> [!NOTE] **[Luogu 南蛮图腾](https://www.luogu.com.cn/problem/P1498)**
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

// https://www.luogu.com.cn/blog/treer/solution-p1498

const int N = 1100;

int n;
int g[N] = {1};

int main() {
    cin >> n;
    for (int i = 0; i < 1 << n; ++i) {
        for (int j = 1; j < (1 << n) - i; ++j)
            cout << " ";  //前导空格
        for (int j = i; j >= 0; --j)
            g[j] ^= g[j - 1];  //修改数组
        if (!(i % 2))
            for (int j = 0; j <= i; ++j)
                cout << (g[j] ? "/\\" : "  ");  //奇数行
        else
            for (int j = 0; j <= i; j += 2)
                cout << (g[j] ? "/__\\" : "    ");  //偶数行
        cout << endl;
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

> [!NOTE] **[Luogu 计算分数](https://www.luogu.com.cn/problem/P1572)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 经典模拟

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    string s;
    cin >> s;
    
    if (s[0] != '-')
        s = "+" + s;
    int n = s.size();
    
    int a = 0, b = 1, f = 1;
    for (int i = 0; i < n; ++ i ) {
        int nf = (s[i] == '+' ? 1 : -1);
        int na = 0, nb = 0;
        int j = i + 1;
        while (isdigit(s[j]))
            na = na * 10 + s[j] - '0', j ++ ;
        j ++ ;  // '/'
        while (isdigit(s[j]))
            nb = nb * 10 + s[j] - '0', j ++ ;
        i = j - 1;  // '+' or '-'
        
        int g = __gcd(b, nb);
        int nnb = b / g * nb;
        int nna = f * nb / g * a + nf * b / g * na;
        
        f = (nna >= 0 ? 1 : -1);
        
        nna = abs(nna);
        g = __gcd(nna, nnb);
        if (g) {
            nna /= g, nnb /= g;
        } else {
            nna = 0, nnb = 1;
        }
        a = nna, b = nnb;
    }
    if (f < 0)
        cout << '-';
    if (a % b == 0)
        cout << to_string(a) << endl;
    else
        cout << to_string(a) << '/' << to_string(b) << endl;
    
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

> [!NOTE] **[LeetCode 592. 分数加减运算](https://leetcode-cn.com/problems/fraction-addition-and-subtraction/)**
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
    string fractionAddition(string expression) {
        int n = 0;
        for (auto c :expression)
            if (c == '/')
                ++ n;
        expression = '+' + expression;
        // 初始 0/1
        int a = 0, b = 1, offset = 0;
        int c, d;
        char e;
        for (int i = 0; i < n; ++ i ) {
            // 对于 +-3/15 e读取+ c读-3 d读15
            // 对于  3/15  e读取+ c读3 d读15
            sscanf(expression.c_str() + offset, "%c%d/%d", &e, &c, &d);
            offset += (e + to_string(c) + '/' + to_string(d)).size();
            if (e == '-') c = -c;
            // x 分子     y 分母
            int x = a * d + b * c, y = b * d;
            int z = __gcd(x, y);
            a = x / z, b = y / z;
        }
        if (b < 0) a = -a, b = -b;
        return to_string(a) + '/' + to_string(b);
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

> [!NOTE] **[LeetCode 68. 文本左右对齐](https://leetcode-cn.com/problems/text-justification/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 一行一行处理，每次先求出这一行最多可以放多少个单词，然后分三种情况处理：
> 
> - 如果是最后一行，则只实现左对齐：每个单词之间插入一个空格，行尾插入若干空格，使这一行的总长度是 maxWidth；
> - 如果这一行只有一个单词，则直接在行尾补上空格；
> - 其他情况，则需计算总共要填补多少空格，然后按题意均分在单词之间；
> 
> 时间复杂度分析：每个单词只会遍历一遍，所以总时间复杂度是 O(n)

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    string space(int x) {
        string res;
        while (x -- ) res += ' ';
        return res;
    }

    vector<string> fullJustify(vector<string>& words, int maxWidth) {
        vector<string> res;
        for (int i = 0; i < words.size();) {
            int j = i + 1, s = words[i].size(), rs = words[i].size();
            while (j < words.size() && s + 1 + words[j].size() <= maxWidth) {
                s += 1 + words[j].size();
                rs += words[j].size();
                j ++ ;
            }
            rs = maxWidth - rs;
            string line = words[i];
            if (j == words.size()) {
                for (i ++; i < j; i ++ )
                    line += ' ' + words[i];
                while (line.size() < maxWidth) line += ' ';
            } else if (j - i == 1) line += space(rs);
            else {
                int base = rs / (j - i - 1);
                int rem = rs % (j - i - 1);
                i ++ ;
                for (int k = 0; i < j; i ++, k ++ )
                    line += space(base + (k < rem)) + words[i];
            }
            i = j;
            res.push_back(line);
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

> [!NOTE] **[LeetCode 71. 简化路径](https://leetcode-cn.com/problems/simplify-path/)**
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
    string simplifyPath(string path) {
        string res, name;
        if (path.back() != '/') path += '/'; // ATTENTION
        for (auto c : path) {
            if (c != '/') name += c;
            else {
                if (name == "..") {
                    while (res.size() && res.back() != '/') res.pop_back();
                    if (res.size()) res.pop_back();
                } else if (name != "." && name != "") {
                    res += '/' + name;
                }
                name.clear();
            }
        }

        if (res.empty()) res = "/";
        return res;
    }
};
```

##### **Python**

```python
"""
path = "/a/./b/../../c/": 
解释： 进入a， '.'表示在当前目录不操作，然后进入b, 然后有两个 '..'，把a 和 b 都弹出来的

我们可以把整个路径看作是一个动态的“进入某个子目录”或“返回上级目录”的过程。
所以我们可以模拟这个过程，用 res 记录当前的路径位置：
1. 如果遇到 ".."，则返回上级目录；
2. 如果遇到 "."或多余的斜杠，则不做任何处理：
3. 其它情况，表示进入某个子目录，我们在 res 后面补上新路径即可；

"""
class Solution:
    def simplifyPath(self, path: str) -> str:
        res = []
        for c in path.split("/"):
            if c not in [".", "..", ""]:
                res.append(c)
            if res and c == "..":
                res.pop()
        return "/" + "/".join(res)
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 80. 删除排序数组中的重复项 II](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array-ii/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 优雅写法

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int removeDuplicates(vector<int>& nums) {
        int k = 0;
        for (auto x : nums)
            if (k < 2 || (nums[k - 1] != x || nums[k - 2] != x))
                nums[k ++ ] = x;
        return k;
    }
};
```

##### **Python**

```python
# 一个指针存储的有效数字，一个指针往后走
# 由于相同的数字可以保存两个，所以可以用当前数和第前2个数相比较

class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 2:return n 
        p1 = 1 
        for p2 in range(2, n):
            if nums[p2] != nums[p1 - 1]:
                p1 += 1
                nums[p1] = nums[p2]
        return p1 + 1
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 149. 直线上最多的点数](https://leetcode-cn.com/problems/max-points-on-a-line/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ new**

```cpp
class Solution {
public:
    int maxPoints(vector<vector<int>>& points) {
        typedef long double LD;

        int res = 0;
        for (auto& p: points) {
            int ss = 0, vs = 0;
            unordered_map<LD, int> cnt;
            for (auto& q: points)
                if (p == q) ss ++ ;
                else if (p[0] == q[0]) vs ++ ;
                else {
                    LD k = (LD)(q[1] - p[1]) / (q[0] - p[0]);
                    cnt[k] ++ ;
                }
            int c = vs;
            for (auto [k, t]: cnt) c = max(c, t);
            res = max(res, c + ss);
        }
        return res;
    }
};
```

##### **C++ old**

```cpp
class Solution {
public:
    int maxPoints(vector<Point>& points) {
        if (points.empty()) return 0;
        int res = 1;
        for (int i = 0; i < points.size(); i ++ ) {
            unordered_map<long double, int> map;
            int duplicates = 0, verticals = 1;

            for (int j = i + 1; j < points.size(); j ++ )
                if (points[i].x == points[j].x) {
                    verticals ++ ;
                    if (points[i].y == points[j].y) duplicates ++ ;
                }

            for (int j = i + 1; j < points.size(); j ++ )
                if (points[i].x != points[j].x) {
                    long double slope = (long double)(points[i].y - points[j].y) / (points[i].x - points[j].x);
                    if (map[slope] == 0) map[slope] = 2;
                    else map[slope] ++ ;
                    res = max(res, map[slope] + duplicates);
                }

            res = max(res, verticals);
        }
        return res;
    }
};
```

##### **C++ abandon**

```cpp
class Solution {
public:
    int maxPoints(vector<vector<int>>& points) {
        int len = points.size();
        if (len < 3) return len;
        int result = 0;
        for (int i = 0; i < len; ++ i ) {
            int duplicate = 0;
            int curMax = 0;
            unordered_map<string, int> oneline;
            for (int j = i+1; j < len; ++ j ) {
                if (points[i][0] == points[j][0] && points[i][1] == points[j][1]) {
                    duplicate += 1;
                    continue;
                }
                int diffX = points[i][0] - points[j][0];
                int diffY = points[i][1] - points[j][1];
                int tmp = gcd(diffX, diffY);
                string key = to_string(diffX/tmp) + "/" + to_string(diffY/tmp);

                oneline[key] ++ ;
                curMax = max(curMax, oneline[key]);
            }
            result = max(result, curMax + duplicate + 1);
        }
        return result;
    }
    int gcd(int a, int b) {
        if (b) return gcd(b, a % b);
        return a;
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

> [!NOTE] **[LeetCode 290. 单词规律](https://leetcode-cn.com/problems/word-pattern/)**
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
    bool wordPattern(string pattern, string str) {
        vector<string> ws;
        string w;
        stringstream ssin(str);
        while (ssin >> w) ws.push_back(w);
        if (pattern.size() != ws.size()) return false;
        unordered_map<char, string> pw;
        unordered_map<string, char> wp;
        for (int i = 0; i < pattern.size(); ++ i ) {
            auto p = pattern[i];
            auto w = ws[i];
            if (pw.count(p) && pw[p] != w) return false;
            pw[p] = w;
            if (wp.count(w) && wp[w] != p) return false;
            wp[w] = p;
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

> [!NOTE] **[LeetCode 498. 对角线遍历](https://leetcode-cn.com/problems/diagonal-traverse/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 优雅遍历方式

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    vector<int> findDiagonalOrder(vector<vector<int>>& w) {
        vector<int> res;
        if (w.empty() || w[0].empty()) return res;
        int n = w.size(), m = w[0].size();
        for (int i = 0; i < n + m -1; ++ i ) {
            if (i % 2 == 0)
                for (int j = min(i, n - 1); j >= max(0, 1 - m + i); -- j )
                    res.push_back(w[j][i - j]);
            else
                for (int j = max(0, 1 - m + i); j <= min(i, n - 1); ++ j )
                    res.push_back(w[j][i - j]);
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

> [!NOTE] **[LeetCode 649. Dota2 参议院](https://leetcode-cn.com/problems/dota2-senate/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 栈 trick**

```cpp
class Solution {
public:
    string predictPartyVictory(string senate) {
        int n = senate.size();
        queue<int> r, d;
        for (int i = 0; i < n; ++ i )
            if (senate[i] == 'R') r.push(i);
            else d.push(i);

        while (r.size() && d.size()) {
            if (r.front() < d.front()) r.push(r.front() + n);
            else d.push(d.front() + n);
            // 思想
            r.pop(), d.pop();
        }
        return r.size() ? "Radiant" : "Dire";
    }
}
```

##### **C++ 纯模拟**

```cpp
class Solution {
public:
    string predictPartyVictory(string senate) {
        set<int> R, D, A;
        int n = senate.size();
        for (int i = 0; i < n; ++ i ) {
            if (senate[i] == 'R') R.insert(i);
            else D.insert(i);
            A.insert(i);
        }
            
        for (;;) {
            for (auto i : A)
                if (senate[i] == 'R') {
                    if (D.empty()) return "Radiant";
                    else {
                        auto t = D.lower_bound(i);
                        if (t == D.end()) t = D.begin();
                        A.erase(*t);
                        D.erase(*t);
                    }
                } else {
                    if (R.empty()) return "Dire";
                    else {
                        auto t = R.lower_bound(i);
                        if (t == R.end()) t = R.begin();
                        A.erase(*t);
                        R.erase(*t);
                    }
                }
        }
        return "";
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

> [!NOTE] **[LeetCode 721. 账户合并](https://leetcode-cn.com/problems/accounts-merge/)**
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
    vector<int> p;
    int find(int x) {
        if (p[x] != x) return p[x] = find(p[x]);
        return p[x];
    }

    vector<vector<string>> accountsMerge(vector<vector<string>>& accounts) {
        int n = accounts.size();
        p.resize(n);
        for (int i = 0; i < n; ++ i ) p[i] = i;

        unordered_map<string, int> rt;
        for (int i = 0; i < n; ++ i ) {
            int sz = accounts[i].size();
            for (int j = 1; j < sz; ++ j )
                if (rt.count(accounts[i][j])) p[find(i)] = p[find(rt[accounts[i][j]])];
                else rt[accounts[i][j]] = i;
        }

        vector<vector<string>> res;
        unordered_map<int, vector<string>> ve;
        for (auto & [e, root] : rt) ve[find(root)].push_back(e);
        for (auto & [root, es] : ve) {
            sort(es.begin(), es.end());
            vector<string> t;
            t.push_back(accounts[root][0]);
            for (auto & e : es) t.push_back(e);
            res.push_back(t);
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

> [!NOTE] **[LeetCode 794. 有效的井字游戏](https://leetcode-cn.com/problems/valid-tic-tac-toe-state/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 优雅实现细节

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    vector<string> g;

    int get(char c) {
        int res = 0;
        for (int i = 0; i < 3; i ++ )
            for (int j = 0; j < 3; j ++ )
                if (g[i][j] == c)
                    res ++ ;
        return res;
    }

    bool check(char c) {
        for (int i = 0; i < 3; i ++ ) {
            if (g[i][0] == c && g[i][1] == c && g[i][2] == c) return true;
            if (g[0][i] == c && g[1][i] == c && g[2][i] == c) return true;
        }
        if (g[0][0] == c && g[1][1] == c && g[2][2] == c) return true;
        if (g[0][2] == c && g[1][1] == c && g[2][0] == c) return true;
        return false;
    }

    bool validTicTacToe(vector<string>& board) {
        g = board;
        bool cx = check('X'), co = check('O');
        if (cx && co) return false;
        int sx = get('X'), so = get('O');
        if (cx && sx != so + 1) return false;
        if (co && sx != so) return false;
        if (sx != so && sx != so + 1) return false;
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

> [!NOTE] **[LeetCode 1152. 用户网站访问行为分析](https://leetcode-cn.com/problems/analyze-user-website-visit-pattern/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 思路正确 **注意本题允许重合** 各种实现小细节

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using PIS = pair<int, string>;
    unordered_map<string, vector<PIS>> hash;
    
    vector<string> mostVisitedPattern(vector<string>& username, vector<int>& timestamp, vector<string>& website) {
        int n = username.size();
        for (int i = 0; i < n; ++ i )
            hash[username[i]].push_back({timestamp[i], website[i]});
        for (auto & [k, v] : hash)
            sort(v.begin(), v.end());
        
        vector<string> ve;
        for (auto & w : website)
            ve.push_back(w);
        sort(ve.begin(), ve.end());
        ve.erase(unique(ve.begin(), ve.end()), ve.end());
        
        int m = ve.size();
        int maxv = 0;
        vector<string> res;
        for (int i = 0; i < m; ++ i )
            for (int j = 0; j < m; ++ j )
                // if (i != j)
                    for (int k = 0; k < m; ++ k ) {
                        // if (j != i && j != k) {
                            int cnt = 0;
                            
                            vector<string> s = {ve[i], ve[j], ve[k]};
                            for (auto & [k, v] : hash) {
                                int p = 0;
                                for (int u = 0; u < v.size() && p < 3; ++ u ) {
                                    auto & [_, web] = v[u];
                                    if (web == s[p])
                                        ++ p ;
                                }
                                if (p == 3)
                                    ++ cnt ;
                            }
                            
                            if (cnt > maxv) {
                                maxv = cnt;
                                res = s;
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


> [!NOTE] **[LeetCode 1153. 字符串转化](https://leetcode-cn.com/problems/string-transforms-into-another-string/)** TAG
> 
> 题意: 注意：原题要求转化为以下条件
> 
> > 如果 `str1 != str2` ，且 `str2` 包含所有的 26 个字母，则不能转化

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
    bool canConvert(string str1, string str2) {
        if (str1 == str2)
            return true;
        unordered_set<char> S;
        for (auto c : str2)
            S.insert(c);
        if (S.size() == 26)
            return false;
        
        int n = str1.size();
        unordered_map<char, char> hash;
        for (int i = 0; i < n; ++ i ) {
            char c1 = str1[i], c2 = str2[i];
            if (!hash.count(c1))
                hash[c1] = c2;
            else if (hash[c1] != c2)
                return false;
        }
        return true;
    }
};
```

##### **C++ 另一思路**

```cpp
class Solution {
public:
    bool canConvert(string s, string t) {
        if (s == t) return true;
        
        int n = s.size();
        vector<int> to(26, -1);
        for (int i = 0; i < n; ++ i) {
            int x = s[i]-'a', y = t[i]-'a';
            if (to[x] == -1) to[x] = y;
            else if (to[x] != y) return false;
        }
        
        // 不是 26 个字符都转换了
        int has = 0;
        for (int i = 0; i < 26; ++ i)
            has += to[i] != -1;
        if (has != 26) return true;
        
        // 转换了 26 个字符 此时需要满足以下条件
        //  存在不同字符转化为同一字符 
        int flag = 0;
        for (int i = 0; i < 26; ++ i)
            for (int j = i+1; j < 26; ++ j)
                if (to[i] != -1 && to[j] != -1 && to[i] == to[j])
                    flag = 1;
        
        return flag;
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

> [!NOTE] **[LeetCode 1409. 查询带键的排列](https://leetcode-cn.com/problems/queries-on-a-permutation-with-key/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 一个排列 每次把某个位置放到第一个 返回操作结束的排列
> 
> 记录对应关系 复杂度比有些操作后还要遍历的好hhh

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    vector<int> processQueries(vector<int>& queries, int m) {
        int len = queries.size();
        vector<int> res;
        unordered_map<int, int> m1, m2;  // m1 idx=>num    m2 num=>idx
        for (int i = 0; i < m; ++i) {
            m1[i] = i + 1;
            m2[i + 1] = i;
        }

        for (int i = 0; i < len; ++i) {
            int num = queries[i];
            int idx = m2[num];
            res.push_back(idx);
            for (int j = idx; j >= 0; --j) {
                m2[m1[j]]++;
                m1[j] = m1[j - 1];
            }
            m1[0] = num;
            m2[num] = 0;
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

> [!NOTE] **[LeetCode 1932. 合并多棵二叉搜索树](https://leetcode-cn.com/problems/merge-bsts-to-create-single-bst/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 非常复杂的模拟 见注释
> 
> 重复做

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
    unordered_map<TreeNode*, TreeNode*> root;   // 找根
    unordered_map<int, TreeNode*> states;       // 某个值作为根节点 是哪个节点
    
    TreeNode* get_root(TreeNode * t) {
        auto it = root.find(t);
        // 自己就是根
        if (it == root.end())
            return t;
        // 自己不是根 继续找
        // 类似于路径压缩的并查集思路
        TreeNode * pa = it->second;
        root[t] = get_root(pa);
        return root[t];
    }
    
    bool is_bst(TreeNode * r, int d, int u) {
        if (!r)
            return true;
        if (r->val < d || r->val > u)
            return false;
        return is_bst(r->left, d, r->val - 1) && is_bst(r->right, r->val + 1, u);
    }
    
    TreeNode* canMerge(vector<TreeNode*>& trees) {
        queue<TreeNode*> q;
        for (auto t : trees) {
            states[t->val] = t; // 
            q.push(t);
        }
        
        while (!q.empty()) {
            auto t = q.front(); q.pop();
            // 加入儿子
            vector<pair<TreeNode*, int>> sons;
            if (t->left)
                sons.push_back({t->left, 0});
            if (t->right)
                sons.push_back({t->right, 1});
            
            for (auto [p, v] : sons) {
                // 非叶子节点
                if (p->left || p->right) {
                    root[p] = t;
                    q.push(p);
                } else {
                    // 叶子节点 找其是否为某个子树根节点
                    auto it = states.find(p->val);
                    if (it == states.end())
                        continue;
                    
                    // CASE: it->second 是某个子树的根节点 且是当前树的根
                    // 则成环 返回nullptr
                    if (it->second == get_root(t))
                        return nullptr;
                    
                    // 更新根信息 使用it替代p所在的位值
                    root[it->second] = t;
                    if (v == 0)
                        t->left = it->second;
                    else
                        t->right = it->second;
                    states.erase(it);   // ATTENTION
                }
            }
        }
        if (states.size() != 1)
            return nullptr;
        TreeNode * rt = states.begin()->second;
        return is_bst(rt, -1e9, 1e9) ? rt : nullptr;
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

> [!NOTE] **[LeetCode 1958. 检查操作是否合法](https://leetcode-cn.com/problems/check-if-move-is-legal/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 任意方向合法均可 主要是理解题意
> 
> 加强写模拟题的熟练度 以加快速度

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    vector<vector<char>> b;
    int n, m;
    int dx[8] = {-1, -1, -1, 0, 1, 1, 1, 0}, dy[8] = {-1, 0, 1, 1, 1, 0, -1, -1};
    
    bool cango(int x, int y, char ch) {
        return x >= 0 && x < n && y >= 0 && y < m && b[x][y] == ch;
    }
    
    bool checkMove(vector<vector<char>>& board, int r, int c, char color) {
        this->b = board;
        this->n = b.size(), m = b[0].size();
        // 根据题意无需此判断
        // if (b[r][c] != '.')
        //     return false;
        
        char ch = (color == 'B' ? 'W' : 'B');
        for (int i = 0; i < 8; ++ i ) {
            int x = r + dx[i], y = c + dy[i], c = 0;
            while (cango(x, y, ch)) {
                c ++ ;
                x += dx[i], y += dy[i];
            }
            if (c > 0 && cango(x, y, color)) {
                return true;
            }
        }
        return false;
    }
};
```

##### **C++ 赛榜**

```cpp
class Solution {
public:
    int dx[8] = { 0, 0, 1, -1, -1, -1, 1, 1 };
    int dy[8] = { 1, -1, 0, 0, 1, -1, 1, -1 };
    bool checkMove(vector<vector<char>>& g, int x, int y, char c) {
        int n = g.size(), m = g[0].size();
        char t = c == 'B' ? 'W' : 'B';
        for (int i = 0; i < 8; i++) {
            int nx = x + dx[i], ny = y + dy[i];
            if (nx < 0 || nx >= n || ny < 0 || ny >= m) continue;
            if (g[nx][ny] == t) {
                while (1) {
                    nx = nx + dx[i], ny = ny + dy[i];
                    if (nx < 0 || nx >= n || ny < 0 || ny >= m) break;
                    if (g[nx][ny] == t) continue;
                    else if (g[nx][ny] == '.') break;
                    else {
                        return 1;
                    }
                }
            }
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

> [!NOTE] **[LeetCode 2018. 判断单词是否能放入填字游戏内](https://leetcode-cn.com/problems/check-if-word-can-be-placed-in-crossword/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 模拟即可 注意筛选条件
> 
> 可逆序细节WA1

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int n, m;
    vector<vector<char>> g;
    
    bool placeWordInCrossword(vector<vector<char>>& board, string word) {
        this->g = board;
        this->n = g.size(), this->m = g[0].size();
        vector<vector<int>> l(n, vector<int>(m)), u(n, vector<int>(m));
        for (int i = 0; i < n; ++ i )
            for (int j = 0; j < m; ++ j )
                if (g[i][j] != '#') {
                    l[i][j] = (j - 1 >= 0 ? l[i][j - 1] : 0) + 1;
                    u[i][j] = (i - 1 >= 0 ? u[i - 1][j] : 0) + 1;
                }
        
        int len = word.size();
        for (int i = 0; i < n; ++ i )
            for (int j = 0; j < m; ++ j ) {
                if (l[i][j] == len && (j == m - 1 || g[i][j + 1] == '#')) {
                    int st = j - len + 1, ed = j;
                    {
                        bool flag = true;
                        for (int k = st; k <= ed; ++ k )
                            if (g[i][k] != ' ' && g[i][k] != word[k - st]) {
                                flag = false;
                                break;
                            }
                        if (flag)
                            return true;
                    }
                    {
                        bool flag = true;
                        for (int k = ed; k >= st; -- k )
                            if (g[i][k] != ' ' && g[i][k] != word[len - (k - st) - 1]) {
                                flag = false;
                                break;
                            }
                        if (flag)
                            return true;
                    }
                }
                if (u[i][j] == len && (i == n - 1 || g[i + 1][j] == '#')) {
                    int st = i - len + 1, ed = i;
                    {
                        bool flag = true;
                        for (int k = st; k <= ed; ++ k )
                            if (g[k][j] != ' ' && g[k][j] != word[k - st]) {
                                flag = false;
                                break;
                            }
                        if (flag)
                            return true;
                    }
                    {
                        bool flag = true;
                        for (int k = ed; k >= st; -- k )
                            if (g[k][j] != ' ' && g[k][j] != word[len - (k - st) - 1]) {
                                flag = false;
                                break;
                            }
                        if (flag)
                            return true;
                    }
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

> [!NOTE] **[Codeforces C. Vasya and Basketball](https://codeforces.com/problemset/problem/493/C)**
> 
> 题意: TODO
> 
> Vasya 记录了一场篮球赛中两支队伍每次命中的投篮离篮筐的距离。他知道每一次成功的投篮可以得到 2 分或 3 分。
> 
> 如果一次命中的投篮离篮筐不超过 $d(d \ge 0)$ 米则得 2 分，否则得 3 分。
> 
> Vasya 可以指定一个 d ，同时他希望第一支队伍的分数 a 减去第二支队伍的分数 b 最大。
> 
> 请你帮他求出这个 dd。

> [!TIP] **思路**
> 
> 写起来细节多很慢
> 
> 思路正确 AC

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: C. Vasya and Basketball
// Contest: Codeforces - Codeforces Round #281 (Div. 2)
// URL: https://codeforces.com/problemset/problem/493/C
// Memory Limit: 256 MB
// Time Limit: 2000 ms
//
// Powered by CP Editor (https://cpeditor.org)

#include <bits/stdc++.h>
using namespace std;

using LL = long long;
using PII = pair<int, int>;
const int N = 200010;

int n, m;
int a[N], b[N];
vector<int> vs;
vector<PII> va, vb;

vector<PII> get(int t[], int n) {
    sort(t, t + n);
    vector<PII> ve;
    int v = 0;
    for (int i = 0; i < n; ++i) {
        if (t[i] != v) {
            // <= v 的有 i 个
            ve.push_back({v, i});
        }
        v = t[i];
    }
    ve.push_back({v, n});
    // for convenience
    ve.push_back({2e9 + 10, n});

    return ve;
}

// n 与 总量 tot 的不同 需要注意 之前细节错了WA
// https://codeforces.com/contest/493/submission/110105139
LL f(vector<PII>& t, int n, int tot, int d) {
    int l = 0, r = n;
    while (l < r) {
        int m = l + r >> 1;
        auto [v, _] = t[m];
        if (v <= d)
            l = m + 1;
        else
            r = m;
    }
    --l;
    auto [_, c] = t[l];
    LL ret = (LL)c * 2 + (LL)(tot - c) * 3;
    return ret;
}

int main() {
    // 可以全部都算3分
    vs.push_back(0);

    cin >> n;
    for (int i = 0; i < n; ++i)
        cin >> a[i], vs.push_back(a[i]);
    va = get(a, n);

    cin >> m;
    for (int i = 0; i < m; ++i)
        cin >> b[i], vs.push_back(b[i]);
    vb = get(b, m);

    sort(vs.begin(), vs.end());
    vs.erase(unique(vs.begin(), vs.end()), vs.end());
    int vsn = vs.size();

    LL av, bv, d = -2e18;
    for (int i = 0; i < vsn; ++i) {
        LL ta = f(va, va.size(), n, vs[i]);
        LL tb = f(vb, vb.size(), m, vs[i]);
        if (ta - tb > d)
            av = ta, bv = tb, d = ta - tb;
    }

    cout << av << ':' << bv << endl;

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

> [!NOTE] **[LeetCode 936. 戳印序列](https://leetcode.cn/problems/stamping-the-sequence/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 逆向模拟 实现细节

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 逆向操作 找到可以匹配的串恢复成 ?
    bool work(string & stamp, string & target, int k) {
        if (target.substr(k, stamp.size()) == string(stamp.size(), '?'))
            return false;
        for (int i = 0; i < stamp.size(); ++ i )
            if (target[k + i] != '?' && stamp[i] != target[k + i])
                return false;
        for (int i = 0; i < stamp.size(); ++ i )
            target[k + i] = '?';
        return true;
    }

    vector<int> movesToStamp(string stamp, string target) {
        vector<int> res;
        for (;;) {
            bool flag = true;
            for (int i = 0; i + stamp.size() <= target.size(); ++ i )
                if (work(stamp, target, i)) {
                    res.push_back(i);
                    flag = false;
                }
            if (flag)
                break;
        }
        if (target != string(target.size(), '?'))
            return {};
        reverse(res.begin(), res.end());
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

### 字符串处理

> [!NOTE] **[LeetCode 388. 文件的最长绝对路径](https://leetcode-cn.com/problems/longest-absolute-file-path/)**
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
    int lengthLongestPath(string input) {
        stack<int> stk;
        int res = 0;
        for (int i = 0, sum = 0; i < input.size(); i ++ ) {
            int k = 0;
            while (i < input.size() && input[i] == '\t') i ++ , k ++ ;
            while (stk.size() > k) sum -= stk.top(), stk.pop();
            int j = i;
            while (j < input.size() && input[j] != '\n') j ++ ;
            int len = j - i;
            stk.push(len), sum += len;
            if (input.substr(i, len).find('.') != -1)
                res = max(res, sum + (int)stk.size() - 1);
            i = j;
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

> [!NOTE] **[LeetCode 393. UTF-8 编码验证](https://leetcode-cn.com/problems/utf-8-validation/)**
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
    int get(int x, int k) {
        return x >> k & 1;
    }
    bool validUtf8(vector<int>& data) {
        for (int i = 0; i < data.size(); ++ i ) {
            // 1 字节
            if (!get(data[i], 7)) continue;
            int k = 0;
            while (k <= 4 && get(data[i], 7 - k)) ++ k ;
            if (k == 1 || k > 4) return false;
            // 此时 k = n + 1 应为 0
            for (int j = 0; j < k - 1; ++ j ) {
                int t = i + 1 + j;
                if (t >= data.size()) return false;
                // 应为 1,0 开头
                if (!(get(data[t], 7) && !get(data[t], 6))) return false;
            }
            i += k - 1;
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

> [!NOTE] **[LeetCode 394. 字符串解码](https://leetcode-cn.com/problems/decode-string/)**
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
    string dfs(string & s, int & u) {
        string res;
        while (u < s.size() && s[u] != ']') {
            if (s[u] >= 'a' && s[u] <= 'z' || s[u] >= 'A' && s[u] <= 'Z') res += s[u ++ ];
            else if (s[u] >= '0' && s[u] <= '9') {
                int k = u;
                while (s[k] >= '0' && s[k] <= '9') ++ k ;
                int x = stoi(s.substr(u, k - u));
                // 左括号
                u = k + 1;
                string y = dfs(s, u);
                u ++ ;  // 过滤右括号
                while (x -- ) res += y;
            }
        }
        return res;
    }
    string decodeString(string s) {
        int u = 0;
        return dfs(s, u);
    }
```

##### **C++ 栈**

```cpp
    string decodeString(string s) {
        string res;
        stack<int> nums;
        stack<string> strs;
        int num = 0, len = s.size();
        for (int i = 0; i < len; ++ i ) {
            if (s[i] >= '0' && s[i] <= '9') num = num * 10 + s[i] - '0';
            else if ((s[i] >= 'a' && s[i] <= 'z') || (s[i] >= 'A' && s[i] <= 'Z')) res.push_back(s[i]);
            else if (s[i] == '[') {
                nums.push(num);
                num = 0;
                strs.push(res);
                res = "";
            } else {
                int times = nums.top();
                nums.pop();
                string top = strs.top();
                strs.pop();
                while (times -- ) top += res;
                res = top;
            }
        }
        return res;
    }
};
```

##### **Python**

```python
# 法一：普通递归
class Solution:
    def decodeString(self, s: str) -> str:
        u = 0
        def dfs(s):
            nonlocal u  # 踩坑1：u需要在递归里 进行变化，并且还在本层 有使用到，所有要用nonlocal
            res = ''
            while u < len(s) and s[u] != ']':
                if s[u].isalpha(): #等价于if 'a' <= s[u] <= 'z' or 'A' <= s[u] <= 'Z':
                    res += s[u]
                    u += 1
                elif s[u].isdigit():  #等价于 if '0' <= s[u] <= '1'
                    k = u
                    while s[k].isdigit():
                        k += 1

                    x = int(s[u:k])
                    u = k + 1#跳过左括号
                    y = dfs(s, u)
                    u += 1 #跳过右括号

                    while x:
                        res += y
                        x -= 1
            return res
        return dfs(s)

#法二：辅助栈法 （ 遇到 “括号”类型的题 都可以想到用 栈 这个数据结构来辅助完成）
# 栈里每次存储两个信息（左括号前的字符串，左括号前的数字）；比如abc3[def], 当遇到第一个左括号的时候，压入栈中的是("abc", 3), 然后遍历括号里面的字符串def, 当遇到右括号的时候, 从栈里面弹出一个元素(s1, n1), 得到新的字符串为s1+n1*"def", 也就是abcdefdefdef。对于括号里面嵌套的情况也是同样处理方式。
# 当遇到 左括号 就进行压栈处理；遇到右括号就弹出栈（栈中记录的元素很重要）
class Solution:
    def decodeString(self, s: str) -> str:
        res, multi, stack = '', 0, []
        for c in s:
            if c.isdigit():    
                multi = multi * 10 + int(c)
            elif c == '[':
                stack.append([res, multi])
                res, multi = '', 0 
            elif c == ']':
                last = stack.pop()
                res = last[0] + last[1] * res 
            else:
                res += c 
        return res
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 443. 压缩字符串](https://leetcode-cn.com/problems/string-compression/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 只留下更好的处理

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int compress(vector<char>& s) {
        int k = 0;
        for (int i = 0; i < s.size(); i ++ ) {
            int j = i + 1;
            while (j < s.size() && s[j] == s[i]) j ++ ;
            int len = j - i;
            s[k ++ ] = s[i];
            if (len > 1) {
                int t = k;
                while (len) {
                    s[t ++ ] = '0' + len % 10;
                    len /= 10;
                }
                reverse(s.begin() + k, s.begin() + t);
                k = t;
            }
            i = j - 1;
        }
        return k;
    }
};
```

##### **Python**

```python
class Solution:
    def compress(self, chars: List[str]) -> int:
        # 重新整理字符串的指针
        k = 0
        i = 0
        while i < len(chars):
            # 相同字符串的右下标
            j = i + 1
            while j < len(chars) and chars[i] == chars[j]:
                j += 1
            # 得到相同字符串的长度
            length = j - i

            # 给头字母赋值
            chars[k] = chars[i]
            k += 1

            # 假如是length > 1的话，说明要往后面填数字
            if length > 1:
                for c in str(length):
                    chars[k] = c
                    k += 1

            # 更新左下标
            i = j

        # 返回我们填完数字的最后一位
        print(k)


```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 467. 环绕字符串中唯一的子字符串](https://leetcode-cn.com/problems/unique-substrings-in-wraparound-string/)**
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
    int findSubstringInWraproundString(string p) {
        // 本质求 p 中有多少个不同子串 使得子串是连续字符
        unordered_map<char, int> cnt;
        for (int i = 0; i < p.size();) {
            int j = i + 1;
            // 相连 一直向右
            while (j < p.size() && (p[j] - p[j - 1] == 1 || p[j] == 'a' && p[j - 1] == 'z')) ++ j ;
            while (i < j) cnt[p[i]] = max(cnt[p[i]], j - i), ++ i ;
        }
        int res = 0;
        for (auto [k, v] : cnt) res += v;
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

> [!NOTE] **[LeetCode 468. 验证IP地址](https://leetcode-cn.com/problems/validate-ip-address/)**
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
    vector<string> split(string ip, char t) {
        vector<string> items;
        for (int i = 0; i < ip.size(); i ++ ) {
            int j = i;
            string item;
            while (ip[j] != t) item += ip[j ++ ];
            i = j;
            items.push_back(item);
        }
        return items;
    }

    string check_ipv4(string ip) {
        auto items = split(ip + '.', '.');
        if (items.size() != 4) return "Neither";
        for (auto item: items) {
            if (item.empty() || item.size() > 3) return "Neither";
            if (item.size() > 1 && item[0] == '0') return "Neither";
            for (auto c: item)
                if (c < '0' || c > '9') return "Neither";
            int t = stoi(item);
            if (t > 255) return "Neither";
        }
        return "IPv4";
    }

    bool check(char c) {
        if (c >= '0' && c <= '9') return true;
        if (c >= 'a' && c <= 'f') return true;
        if (c >= 'A' && c <= 'F') return true;
        return false;
    }

    string check_ipv6(string ip) {
        auto items = split(ip + ':', ':');
        if (items.size() != 8) return "Neither";
        for (auto item: items) {
            if (item.empty() || item.size() > 4) return "Neither";
            for (auto c: item)
                if (!check(c)) return "Neither";
        }
        return "IPv6";
    }

    string validIPAddress(string ip) {
        if (ip.find('.') != -1 && ip.find(':') != -1) return "Neither";
        if (ip.find('.') != -1) return check_ipv4(ip);
        if (ip.find(':') != -1) return check_ipv6(ip);
        return "Neither";
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

> [!NOTE] **[LeetCode 605. 种花问题](https://leetcode-cn.com/problems/can-place-flowers/)**
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
    bool canPlaceFlowers(vector<int>& flowerbed, int n) {
        //if (!n) return true;
        int res = 0;
        for (int i = 0; i < flowerbed.size(); ++ i ) {
            if (flowerbed[i]) continue;
            int j = i;
            while (j < flowerbed.size() && !flowerbed[j]) ++ j;
            // 连续0的长度
            int k = j - i - 1;
            if (!i) ++ k;
            if (j == flowerbed.size()) ++ k;
            res += k / 2;
            if (res >= n) return true;
            i = j;  // j == f.size() || flowerbed[i] = 1
        }
        return res >= n;
    }
};
```

##### **Python**

```python
#一定是在连续若干个0里填入1；找规律：
#思路1：1）如果当前及左右三个位置中存在1，那么跳过；否则就可以种花，种完花n减1；即flowerbed[i]=1;n-=1
#2) 当n==0时，说明已经种完花了，返回True；3）如果最后不满足上述条件，就返回False

class Solution:
    def canPlaceFlowers(self, flowerbed: List[int], n: int) -> bool:
        if n == 0:return True
        m = len(flowerbed)
        for i in range(m):
             # 判断当前及左右位置是否种植了花
            if flowerbed[i] == 1: continue
            if i > 0 and flowerbed[i - 1] == 1: continue
            if i + 1 < m and flowerbed[i + 1] == 1: continue
            flowerbed[i] = 1
            n -= 1
            #踩坑：在这里就需要判断n是否为0，因为存在“[0,0,1,0,0]---1” 这一类的case，可以放入的1个数大于题目给的n
            if n == 0:
                return True
        return False
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 640. 求解方程](https://leetcode-cn.com/problems/solve-the-equation/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 线性处理 实现

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    pair<int, int> work(string s) {
        int a = 0, b = 0;
        if (s[0] != '+' && s[0] != '-') s = '+' + s;
        for (int i = 0; i < s.size(); ++ i ) {
            int j = i + 1;
            while (j < s.size() && isdigit(s[j])) ++ j ;
            int c = 1;
            if (i + 1 <= j - 1) c = stoi(s.substr(i + 1, j - 1 - i));
            if (s[i] == '-') c = -c;
            if (j < s.size() && s[j] == 'x') {
                // 系数项
                a += c;
                i = j;
            } else {
                // 常数项
                b += c;
                i = j - 1;
            }
        }
        return {a, b};
    }

    // ax + b == cx + d pair返回ab cd
    string solveEquation(string equation) {
        int k = equation.find('=');
        auto left = work(equation.substr(0, k)), right = work(equation.substr(k + 1));
        int a = left.first - right.first, b = right.second - left.second;
        if (!a) {
            if (!b) return "Infinite solutions";
            else return "No solution";
        }
        return "x=" + to_string(b / a);
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

> [!NOTE] **[LeetCode 722. 删除注释](https://leetcode-cn.com/problems/remove-comments/)**
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
    vector<string> removeComments(vector<string>& source) {
        string str;
        for (auto& s: source) str += s + '\n';
        vector<string> res;
        string line;

        int n = str.size();
        for (int i = 0; i < n; ) {
            if (i + 1 < n && str[i] == '/' && str[i + 1] == '/') {
                while (str[i] != '\n') i ++ ;
            } else if (i + 1 < n && str[i] == '/' && str[i + 1] == '*') {
                i += 2;
                while (str[i] != '*' || str[i + 1] != '/') i ++ ;
                i += 2;
            } else if (str[i] == '\n') {
                if (line.size()) {
                    res.push_back(line);
                    line.clear();
                }
                i ++ ;
            } else {
                line += str[i];
                i ++ ;
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

> [!NOTE] **[LeetCode 767. 重构字符串](https://leetcode-cn.com/problems/reorganize-string/)**
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
    string reorganizeString(string S) {
        int n = S.size();
        vector<int> cnt(26);
        for (auto c : S) {
            cnt[c - 'a'] += 100;
            if (cnt[c - 'a'] / 100 > (n + 1) / 2) return "";
        }
        // 加入字符信息
        for (int i = 0; i < 26; ++ i ) cnt[i] += i;
        sort(cnt.begin(), cnt.end());
        // 【先填偶数坑，再填奇数坑l。因为对于某个长度为计数且个数恰好为(len+1)/2的字符 在后面填写的时候会撞】
        int idx = 1;
        string res(n, ' ');
        for (auto v : cnt) {
            int sz = v / 100;
            char c = 'a' + v % 100;
            while (sz -- ) {
                if (idx >= n) idx = 0;
                res[idx] = c;
                idx += 2;
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
    string reorganizeString(string S) {
        unordered_map<char, int> cnt;
        int maxc = 0;
        for (auto c: S) {
            cnt[c] ++ ;
            maxc = max(maxc, cnt[c]);
        }
        int n = S.size();
        if (maxc > (n + 1) / 2) return "";
        string res(n, ' ');
        int i = 1, j = 0;
        for (char c = 'a'; c <= 'z'; c ++ ) {
            if (cnt[c] <= n / 2) {
                while (cnt[c] && i < n) {
                    res[i] = c;
                    cnt[c] -- ;
                    i += 2;
                }
            }
            while (cnt[c] && j < n) {
                res[j] = c;
                cnt[c] -- ;
                j += 2;
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

> [!NOTE] **[LeetCode 1169. 查询无效交易](https://leetcode-cn.com/problems/invalid-transactions/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 暴力即可 cpp 处理会麻烦些

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using TSIIS = tuple<string, int, int, string>;
    
    tuple<string, int, int, string> get(string & tx) {
        int sz = tx.size();
        string name, city;
        int time, amount, id = 0;
        for (int i = 0; i < sz; ++ i ) {
            int j = i + 1;
            while (j < sz && tx[j] != ',') ++ j ;
            ++ id;
            string sub = tx.substr(i, j - i);
            if (id == 1) name = sub;
            else if (id == 2) time = stoi(sub.c_str());
            else if (id == 3) amount = stoi(sub.c_str());
            else city = sub;
            i = j;
        }
        return {name, time, amount, city};
    }
    
    vector<string> invalidTransactions(vector<string>& transactions) {
        int n = transactions.size();
        vector<TSIIS> txs;
        for (int i = 0; i < n; ++ i ) txs.push_back(get(transactions[i]));
        
        vector<bool> neg(n, false);
        for (int i = 0; i < n; ++ i ) {
            auto & [name, time, amount, city] = txs[i];
            if (amount > 1000) neg[i] = true;   // can not 'continue'
            for (int j = 0; j < i; ++ j ) {
                auto & [na, t, a, c] = txs[j];
                if (na == name && abs(t - time) <= 60 && c != city)
                    neg[i] = neg[j] = true;
            }
        }
        vector<string> res;
        for (int i = 0; i < n; ++ i )
            if (neg[i])
                res.push_back(transactions[i]);
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

> [!NOTE] **[LeetCode 1209. 删除字符串中的所有相邻重复项 II](https://leetcode-cn.com/problems/remove-all-adjacent-duplicates-in-string-ii/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 有基于栈更简洁的写法

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using PCI = pair<char, int>;
    vector<PCI> get(string s) {
        int n = s.size();
        char ch = s[0];
        int cnt = 1;
        vector<PCI> ve;
        for (int i = 1; i < n; ++ i ) {
            if (s[i] == ch) ++ cnt;
            else {
                ve.push_back({ch, cnt});
                ch = s[i];
                cnt = 1;
            }
        }
        ve.push_back({ch, cnt});
        return ve;
    }
    string removeDuplicates(string s, int k) {
        for (;;) {
            bool f = false;
            auto ve = get(s);
            int sz = ve.size();
            for (int i = 0; i < sz; ++ i ) {
                auto & [ch, cnt] = ve[i];
                if (cnt < k) continue;
                cnt %= k;
                
                f = true;
            }
            
            s = "";
            for (auto [ch, cnt] : ve)
                if (cnt) {
                    string t = string(cnt, ch);
                    s += t;
                }
            
            if (!f) break;
        }
        return s;
    }
};
```

##### **C++ 栈**

```cpp
class Solution {
public:
    using pii = pair<int, int>;
    string removeDuplicates(string s, int k) {
        vector<pii> q;
        q.push_back({'A', 0});
        for (auto c : s) {
            if (c == q.back().first) {
                q.back().second ++;
                if (q.back().second == k) q.pop_back();
            }
            else {
                q.push_back({c, 1});
            }
        }
        string ret;
        for (auto [x, y] : q)
            for (int i = 0; i < y; ++ i) ret += x;
        return ret;
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

> [!NOTE] **[LeetCode 1410. HTML 实体解析器](https://leetcode-cn.com/problems/html-entity-parser/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 从后向前遍历
> 
> TODO 更优雅的实现

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    string entityParser(string text) {
        int len = text.size();
        string res, tmp;
        for (int i = len - 1; i >= 0; --i) {
            if (text[i] == ';') {
                if (i - 5 >= 0 && text[i - 1] == 't' && text[i - 2] == 'o' &&
                    text[i - 3] == 'u' && text[i - 4] == 'q' &&
                    text[i - 5] == '&') {
                    if (tmp.size() > 0) {
                        res += tmp;
                        tmp = "";
                    }
                    res.push_back('"');
                    i -= 5;
                } else if (i - 5 >= 0 && text[i - 1] == 's' &&
                           text[i - 2] == 'o' && text[i - 3] == 'p' &&
                           text[i - 4] == 'a' && text[i - 5] == '&') {
                    if (tmp.size() > 0) {
                        res += tmp;
                        tmp = "";
                    }
                    res.push_back('\'');
                    i -= 5;
                } else if (i - 4 >= 0 && text[i - 1] == 'p' &&
                           text[i - 2] == 'm' && text[i - 3] == 'a' &&
                           text[i - 4] == '&') {
                    if (tmp.size() > 0) {
                        res += tmp;
                        tmp = "";
                    }
                    res.push_back('&');
                    i -= 4;
                } else if (i - 3 >= 0 && text[i - 1] == 't' &&
                           text[i - 2] == 'g' && text[i - 3] == '&') {
                    if (tmp.size() > 0) {
                        res += tmp;
                        tmp = "";
                    }
                    res.push_back('>');
                    i -= 3;
                } else if (i - 3 >= 0 && text[i - 1] == 't' &&
                           text[i - 2] == 'l' && text[i - 3] == '&') {
                    if (tmp.size() > 0) {
                        res += tmp;
                        tmp = "";
                    }
                    res.push_back('<');
                    i -= 3;
                } else if (i - 6 >= 0 && text[i - 1] == 'l' &&
                           text[i - 2] == 's' && text[i - 3] == 'a' &&
                           text[i - 4] == 'r' && text[i - 5] == 'f' &&
                           text[i - 6] == '&') {
                    if (tmp.size() > 0) {
                        res += tmp;
                        tmp = "";
                    }
                    res.push_back('/');
                    i -= 6;
                } else
                    tmp.push_back(text[i]);
            } else
                tmp.push_back(text[i]);
        }
        if (tmp.size() > 0) res += tmp;
        reverse(res.begin(), res.end());
        return res;
    }
};
// 双引号：字符实体为 &quot; ，对应的字符是 " 。
// 单引号：字符实体为 &apos; ，对应的字符是 ' 。
// 与符号：字符实体为 &amp; ，对应对的字符是 & 。
// 大于号：字符实体为 &gt; ，对应的字符是 > 。
// 小于号：字符实体为 &lt; ，对应的字符是 < 。
// 斜线号：字符实体为 &frasl; ，对应的字符是 / 。
```

##### **C++ 优雅实现**

```cpp

```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 1432. 改变一个整数能得到的最大差值](https://leetcode-cn.com/problems/max-difference-you-can-get-from-changing-an-integer/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 数字替换 不能有前导0 求两次替换最大差值
> 
> 题解很多遍历的 太麻烦了

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int maxDiff(int num) {
        string nums = to_string(num);
        int n = nums.size();
        string a = nums, b = nums;
        // 求最大数就直接换9即可
        for (int i = 0; i < n; ++i) {
            if (a[i] != '9') {
                char c = a[i];
                for (int j = i; j < n; ++j) {
                    if (a[j] == c) a[j] = '9';
                }
                break;
            }
        }
        char first = b[0];
        // 求最小数 如果第一位不是1就全换1
        // 否则找到个不是0的(且不是第一位值的)后面全换0
        for (int i = 0; i < n; ++i) {
            if (!i && b[0] != '1') {
                for (int j = 0; j < n; ++j)
                    if (b[j] == first) b[j] = '1';
                break;
            } else if (i > 0 && b[i] != '0' && b[i] != first) {
                char c = b[i];
                for (int j = 0; j < n; ++j)
                    if (b[j] == c) b[j] = '0';
                break;
            }
        }
        return stoi(a) - stoi(b);
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

> [!NOTE] **[LeetCode 1839. 所有元音按顺序排布的最长子字符串](https://leetcode-cn.com/problems/longest-substring-of-all-vowels-in-order/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 线性扫描即可 滑动窗口也可
> 
> 注意：如果不满足字典序 进阶写法

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 线性扫描**

```cpp
class Solution {
public:
    int longestBeautifulSubstring(string word) {
        int inf = 500000;
        int A = -inf, B = -inf, C = -inf, D = -inf, E = -inf, ans = 0;
        for(char c : word){
            if(c == 'a'){
                A = max(A + 1, 1);
                B = C = D = E = -inf;
            }
            if(c == 'e'){
                B = max(A + 1, B + 1);
                A = C = D = E = -inf;
            }
            if(c == 'i'){
                C = max(B + 1, C + 1);
                A = B = D = E = -inf;
            }
            if(c == 'o'){
                D = max(D + 1, C + 1);
                A = B = C = E = -inf;
            }
            if(c == 'u'){
                E = max(E + 1, D + 1);
                A = B = C = D = -inf;
            }
            ans = max(ans, E);
        }
        return ans;
    }
};
```

##### **C++ 滑动窗口**

```cpp
class Solution {
public:
    unordered_map<char, int> hash;
    
    bool check() {
        int c = 0;
        for (auto [k, v] : hash)
            if (v > 0)
                c ++ ;
        return c == 5;
    }
    
    int longestBeautifulSubstring(string word) {
        int n = word.size();
        int res = 0;
        for (int i = 0; i < n; ++ i ) {
            hash.clear();  // 注意清空
            int j = i;
            while (j < n && (j == i || word[j] >= word[j - 1]))
                hash[word[j]] ++ , j ++ ;
            if (check())
                res = max(res, j - i);
            i = j - 1;
        }
        return res;
    }
};
```

##### **C++ 进阶**

```cpp
class Solution {
public:
    int longestBeautifulSubstring(string s) {
        int res = 0;
        string p = "aeiou";
        for (int i = 0; i < s.size(); i ++ ) {
            if (s[i] != 'a') continue;
            int j = i, k = 0;
            while (j < s.size()) {
                if (s[j] == p[k]) j ++ ;
                else {
                    if (k == 4) break;
                    if (s[j] == p[k + 1]) j ++, k ++ ;
                    else break;
                }
                if (k == 4) res = max(res, j - i);
            }
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

> [!NOTE] **[LeetCode 1849. 将字符串拆分为递减的连续值](https://leetcode-cn.com/problems/splitting-a-string-into-descending-consecutive-values/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ ULL**

```cpp
class Solution {
public:
    using ULL = unsigned long long;
    int n;
    string s;
    
    bool check(int p) {
        ULL v = stoull(s.substr(0, p));
        int r = p;
        while (r < n) {
            ULL t = 0;
            int q = r;
            while (q < n && t * 10 + s[q] - '0' <= v - 1) {
                t = t * 10 + s[q] - '0';
                q ++ ;
            }
            if (q == r || t != v - 1)
                return false;
            r = q;
            v = t;
        }
        // return r == n;
        return true;
    }
    
    bool splitString(string s) {
        n = s.size();
        this->s = s;
        if (n == 1)
            return false;
        
        for (int i = 1; i < n; ++ i )
            if (check(i))
                return true;
        return false;
    }
};
```

##### **C++ 状压枚举**

```cpp
class Solution {
public:
    using ULL = unsigned long long;
    
    bool splitString(string s) {
        int n = s.size();
        for (int i = 1; i < 1 << n - 1; ++ i ) {
            bool f = true;
            ULL last = -1, x = s[0] - '0';
            for (int j = 0; j < n - 1; ++ j )
                if (i >> j & 1) {
                    if (last != -1 && x != last - 1) {
                        f = false;
                        break;
                    }
                    last = x, x = s[j + 1] - '0';
                } else
                    x = x * 10 + s[j + 1] - '0';
            if (x != last - 1)
                f = false;
            if (f)
                return true;
        }
        return false;
    }
};
```

##### **C++ 递归**

```cpp
class Solution {
public:
    bool dfs(string s, long long prev, int u) {
        if (u >= s.size()) return true;
        typedef long long LL;
        for (int i = 1; u + i <= s.size(); ++ i) {
            string str = s.substr(u, i);
            LL t = stoll(str);
            if (t > 1e11) return false;
            if (prev != -1 && t > prev - 1) return false;
            if (prev == -1 && i != s.size() || prev - 1 == t) {
                if (dfs(s, t, u + i)) return true;
            }
        }
        return false;
    }
    bool splitString(string s) {
        return dfs(s, -1, 0);
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

### 丑数

> [!NOTE] **[LeetCode 263. 丑数](https://leetcode-cn.com/problems/ugly-number/)**
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
    bool isUgly(int num) {
        if (num <= 0) return false;
        while (num % 2 == 0) num /= 2;
        while (num % 3 == 0) num /= 3;
        while (num % 5 == 0) num /= 5;
        return num == 1;
    }
};
```

##### **Python**

```python
class Solution:
    def isUgly(self, n: int) -> bool:
        if n == 0:return False
        for p in 2, 3, 5:
            while n and n % p == 0:
                n //= p
        return n == 1
      

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 264. 丑数 II](https://leetcode-cn.com/problems/ugly-number-ii/)**
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
    int nthUglyNumber(int n) {
        vector<int> q(1, 1);
        for (int i = 0, j = 0, k = 0; q.size() < n;) {
            int t = min(q[i] * 2, min(q[j] * 3, q[k] * 5));
            q.push_back(t);
            if (q[i] * 2 == t) i ++ ;
            if (q[j] * 3 == t) j ++ ;
            if (q[k] * 5 == t) k ++ ;
        }
        return q.back();
    }
};
```

##### **Python**

```python
class Solution:
    def nthUglyNumber(self, n: int) -> int:
        nums = [2, 3, 5]
        m = len(nums)
        p = [0] * m
        f = [1] * n 
        for i in range(1, n):
            f[i] = min(x * f[y] for x, y in zip(nums, p))
            for j in range(m):
                if f[i] == nums[j] * f[p[j]]:
                    p[j] += 1
        return f[-1]
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 313. 超级丑数](https://leetcode-cn.com/problems/super-ugly-number/)**
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
    int nthSuperUglyNumber(int n, vector<int>& primes) {
        set<int> res={1};
        vector<set<int>::iterator> points;
        int k = primes.size();
        for (int i = 0; i < k; ++ i ) points.push_back(res.begin());
        while (res.size() < n) {
            int temp = INT_MAX;
            // 找最小数
            for (int i = 0; i < k; ++ i )
                temp = min(temp, *points[i] * primes[i]);
            res.insert(temp);
            // 更新指针 此时可能有多个指针都更新了(计算出的值相同的情况)
            for (int i = 0; i < k; ++ i )
                if (temp == *points[i] * primes[i])
                    ++ points[i] ;
        }
        return *res.rbegin();
    }
};

// yxc
// 在 k 较大的时候效果较好
class Solution {
public:
    int nthSuperUglyNumber(int n, vector<int>& primes) {
        typedef pair<int, int> PII;
        priority_queue<PII, vector<PII>, greater<PII>> heap;
        for (int x: primes) heap.push({x, 0});
        vector<int> q(n);
        q[0] = 1;
        for (int i = 1; i < n;) {
            auto t = heap.top(); heap.pop();
            if (t.first != q[i - 1]) q[i ++ ] = t.first;
            int idx = t.second, p = t.first / q[idx];
            heap.push({p * q[idx + 1], idx + 1});
        }
        return q[n - 1];
    }
};
```

##### **Python**

```python
"""
动态规划： 记录前面的丑数，根据每个质因数当前对应的丑数进行乘积大小对比，将最小的那个作为新的丑数，并更新对应的丑数。
"""
class Solution:
    def nthSuperUglyNumber(self, n: int, primes: List[int]) -> int:
        m = len(primes)
        # dp[i] 代表第i+1个丑数
        dp = [inf] * n
        dp[0] = 1
        # indexes代表每个质因子现在应该跟哪个丑数相乘
        indexes = [0] * m

        for i in range(1, n):
            # 哪个质因子相乘的丑数将会变化
            changeIndex = 0
            for j in range(m):
                # 如果当前质因子乘它的丑数小于当前的丑数，更新当前丑数并更新变化坐标
                if primes[j] * dp[indexes[j]] < dp[i]:
                    changeIndex = j
                    dp[i] = primes[j] * dp[indexes[j]]
                # 如果相等直接变化，这样可以去重复
                elif primes[j] * dp[indexes[j]] == dp[i]:
                    indexes[j] += 1
            # 变化的坐标+1
            indexes[changeIndex] += 1
        return dp[-1]
```

<!-- tabs:end -->
</details>

<br>

* * *

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
/*
1. 按两次等效于没按
2. 按按钮的顺序是无关的

则 最多可以达到的状态数量只有16种

按 2 + 3 == 1
按 1 + 2 == 3
按 1 + 3 == 2

*/
#include <bits/stdc++.h>
using namespace std;

int state[8][6] = {
    {1, 1, 1, 1, 1, 1}, // wu
    {0, 0, 0, 0, 0, 0}, // 1
    {0, 1, 0, 1, 0, 1}, // 2
    {1, 0, 1, 0, 1, 0}, // 3
    {0, 1, 1, 0, 1, 1}, // 4
    {1, 0, 0, 1, 0, 0}, // 14
    {1, 1, 0, 0, 0, 1}, // 24
    {0, 0, 1, 1, 1, 0}, // 34
};

int n, c;
vector<int> on, off;

bool check(int s[6]) {
    for (auto id : on)
        if (!s[id % 6])
            return false;
    for (auto id : off)
        if (s[id % 6])
            return false;
    return true;
}

void work(vector<int> ids) {
    sort(ids.begin(), ids.end(), [](int a, int b) {
        for (int i = 0; i < 6; ++ i )
            if (state[a][i] != state[b][i])
                return state[a][i] < state[b][i];
        return false;
    });
    
    bool has_print = false;
    for (auto id : ids) {
        auto s = state[id];
        if (check(s)) {
            has_print = true;
            for (int i = 0; i < n; ++ i )
                cout << s[i % 6];
            cout << endl;
        }
    }
    if (!has_print) cout << "IMPOSSIBLE" << endl;
}

int main() {
    cin >> n >> c;
    int id;
    while (cin >> id, id != -1) on.push_back(id - 1);
    while (cin >> id, id != -1) off.push_back(id - 1);
    
    if (!c) work({0});
    else if (c == 1) work({1, 2, 3, 4});
    else if (c == 2) work({0, 1, 2, 3, 5, 6, 7});
    else work({0, 1, 2, 3, 4, 5, 6, 7});
    
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

> [!NOTE] **[LeetCode 1138. 字母板上的路径](https://leetcode-cn.com/problems/alphabet-board-path/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> **也可以巧妙 trick 简化对于 z 的特殊处理**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

注意不能越界访问即可 略

```cpp
class Solution {
public:
    using PII = pair<int, int>;
    const static int N = 6;
    
    int b[N][N];
    unordered_map<int, PII> mp;
    
    void init() {
        memset(b, -1, sizeof b);
        for (int i = 0, k = 0; i < 6 && k < 26; ++ i )
            for (int j = 0; j < 5 && k < 26; ++ j , ++ k )
                b[i][j] = k, mp[k] = {i, j};
    }
    
    bool check(int x, int y) {
        return x >= 0 && x < 6 && y >= 0 && y < 5 && x * 5 + y < 26;
    }
    
    // ATTENTION 不能出界
    string get(int x, int y, int nx, int ny) {
        string ret;
        while (x != nx || y != ny) {
            if (x != nx) {
                if (nx - x > 0) {
                    char c = 'D';
                    while (x != nx && check(x + 1, y)) {
                        ret.push_back(c);
                        x ++ ;
                    }
                } else {
                    char c = 'U';
                    while (x != nx && check(x - 1, y)) {
                        ret.push_back(c);
                        x -- ;
                    }
                }
            }
            if (y != ny) {
                if (ny - y > 0) {
                    char c = 'R';
                    while (y != ny && check(x, y + 1)) {
                        ret.push_back(c);
                        y ++ ;
                    }
                } else {
                    char c = 'L';
                    while (y != ny && check(x, y - 1)) {
                        ret.push_back(c);
                        y -- ;
                    }
                }
            }
        }
        
        ret.push_back('!');
        return ret;
    }
    
    string alphabetBoardPath(string target) {
        init();
        string res;
        int x = 0, y = 0;
        for (auto c : target) {
            auto [nx, ny] = mp[c - 'a'];
            res += get(x, y, nx, ny);
            x = nx, y = ny;
        }
        
        return res;
    }
};
```

##### **C++ trick**

```cpp
class Solution {
public:
    string alphabetBoardPath(string target) {
        // 模拟 注意‘z’必须先左后下，先上后右
        string res;
        int x = 0, y = 0;
        for (char c : target) {
            int t = c - 'a', nx = t / 5, ny = t % 5;
            if (ny < y)
                for (int _ = 0; _ < y - ny; _ ++)
                    res += 'L';
            if (nx > x)
                for (int _ = 0; _ < nx - x; _ ++)
                    res += 'D';
            if (nx < x)
                for (int _ = 0; _ < x - nx; _ ++)
                    res += 'U';
            if (ny > y)
                for (int _ = 0; _ < ny - y; _ ++)
                    res += 'R';
            res += '!';
            x = nx, y = ny;
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

> [!NOTE] **[LeetCode 1887. 使数组元素相等的减少操作次数](https://leetcode-cn.com/problems/reduction-operations-to-make-the-array-elements-equal/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 从最大的开始即可
> 
> 有多种代码实现

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 裸**

```cpp
class Solution {
public:
    using LL = long long;
    using PII = pair<int, int>;
    
    int reductionOperations(vector<int>& nums) {
        map<int, int> hash;
        for (auto & v : nums)
            hash[v] ++ ;
        
        vector<PII> ve;
        for (auto & [k, v] : hash)
            ve.push_back({k, v});
        sort(ve.begin(), ve.end());
        
        int n = ve.size();
        LL res = 0, c = 0;
        for (int i = n - 1; i >= 0; -- i ) {
            res += c;
            c += ve[i].second;
        }
        
        return res;
    }
};
```

##### **C++ 简化**

```cpp
class Solution {
public:
    int reductionOperations(vector<int>& nums) {
        map<int, int, greater<int>> mp;
        for (int x : nums) mp[x] += 1;
        int ans = 0, sum = 0;
        for (auto& [x, y] : mp){
            ans += sum;
            sum += y;
        }
        return ans;
    }
};
```

##### **C++ 另一思路**

```cpp
class Solution {
public:
    int reductionOperations(vector<int>& a) {
        sort(a.begin(), a.end());
        int res = 0;
        for (int i = 1, s = 0; i < a.size(); i ++ ) {
            if (a[i] != a[i - 1]) s ++ ;
            res += s;
        }
        return res;
    }
};
```

##### **Python**

```python
# 每个数 变动的次数 等于 比小于他的数的个数；
# 从前到后扫描，记录一下当前一共有多少个数 比当前数小。

# 哈希表 + 遍历
class Solution:
    def reductionOperations(self, nums: List[int]) -> int:
        import collections
        my_cnt = collections.Counter(nums)
        ve = []
        for k, v in my_cnt.items():
            ve.append([k,v])
        ve.sort()
        n = len(ve)
        res = 0; c = 0
        for i in range(n-1, -1, -1):
            res += c 
            c += ve[i][1]
        return res
      
# 更简单的写法：
class Solution:
    def reductionOperations(self, nums: List[int]) -> int:
        nums.sort()
        res = 0
        s = 0
        for i in range(1, len(nums)):
            if nums[i] != nums[i-1]:
                s += 1
            res += s 
        return res
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[Codeforces Ciel and Robot](http://codeforces.com/problemset/problem/321/A)**
> 
> 题意: 
> 
> 你现在在一个迷宫的 $(0,0)$ 处，给定一个包含 $\texttt{U,D,L,R}$ 的操作序列 $s$ 
> 
> 其中 $\texttt{U}$ 表示向上走一格，$\texttt{D}$ 表示向下走一格，$\texttt{L}$ 表示向左走一格，$\texttt{R}$ 表示向右走一格。
> 
> 你将会按照 $s$ 从左往右的操作移动，并且重复若干次。问你是否能够到达 $(a,b)$ 处。

> [!TIP] **思路**
> 
> 非常经典的模拟
> 
> 1. 最终可能不是一个完整的循环，故枚举走到哪一步
> 
> 2. 再判断去除这部分后剩下的能否走 d 个完整循环
>    
>    - d >= 0 且横纵坐标相同
>    
>    - 再次计算验证可以到达目标点

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: A. Ciel and Robot
// Contest: Codeforces - Codeforces Round #190 (Div. 1)
// URL: https://codeforces.com/problemset/problem/321/A
// Memory Limit: 256 MB
// Time Limit: 1000 ms

#include <bits/stdc++.h>
using namespace std;

using PII = pair<int, int>;

int a, b;
string s;

int main() {
    cin >> a >> b >> s;

    int n = s.size();
    vector<PII> xs;
    {
        int x = 0, y = 0;
        xs.push_back({0, 0});  // 可以移动之前判断
        for (auto c : s) {
            if (c == 'U')
                y++;
            else if (c == 'D')
                y--;
            else if (c == 'L')
                x--;
            else if (c == 'R')
                x++;
            xs.push_back({x, y});
        }
    }
    auto [sx, sy] = xs.back();

    for (int i = 0; i < xs.size(); ++i) {
        auto [dx, dy] = xs[i];
        int d1 = 2e9, d2 = 2e9;
        if (sx)
            d1 = (a - dx) / sx;
        if (sy)
            d2 = (b - dy) / sy;

        if (d1 < 0 || d2 < 0)
            continue;
        if (d1 != 2e9 && d2 != 2e9 && d1 != d2)
            continue;

        int nx = 0, ny = 0;
        if (d1 != 2e9)
            nx = d1 * sx;
        if (d2 != 2e9)
            ny = d2 * sy;

        if (nx + dx == a && ny + dy == b) {
            cout << "Yes" << endl;
            return 0;
        }
    }
    cout << "No" << endl;

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

> [!NOTE] **[Codeforces No to Palindromes!](http://codeforces.com/problemset/problem/464/A)**
> 
> 题意: 
> 
> 给你一个长度为N的串，其中只包含前 p 个英文字母，保证输入的是一个子序列中没有长度为 2 或者是长度大于 2 的回文串。
> 
> 让你找到一个比原字符串字典序大的第一个也满足: 子序列中没有长度为2或者是长度大于 2 的回文串的解。如果不存在，输出NO。

> [!TIP] **思路**
> 
> 为了让下一次循环可以修改这一位，可以把这一位后面的所有位都置为最大值（即 `'a'+p-1` 这样下一次就可以通过进位直接处理不符合要求的字符

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
// Problem: A. No to Palindromes!
// Contest: Codeforces - Codeforces Round #265 (Div. 1)
// URL: https://codeforces.com/problemset/problem/464/A
// Memory Limit: 256 MB
// Time Limit: 1000 ms

#include <bits/stdc++.h>
using namespace std;

// 为了字典序最小显然要从后往前修改
// 修改之后，只需要关注之前的连续两个位置就可以（因为原串就是合法的）

int n, p;
string str;

int main() {
    cin >> n >> p;
    cin >> str;
    str = ' ' + str;

    bool flag = false;
    for (;;) {
        int j = n;
        str[j]++;
        // 处理进位
        while (j > 1 && str[j] >= 'a' + p)
            str[j] -= p, str[--j]++;
        if (j == 1 && str[j] >= 'a' + p)
            break;  // flag = false;

        // 从前往后检查
        bool fail = false;
        for (int i = j; i <= n; ++i) {
            if (i > 2 && str[i] == str[i - 2]) {
                fail = true;
                break;
            }
            if (i > 1 && str[i] == str[i - 1]) {
                fail = true;
                break;
            }
        }
        if (fail)
            // why? 这样方便下次直接从更前面修改
            for (int i = j + 1; i <= n; ++i)
                str[i] = 'a' + p - 1;
        else {
            flag = true;
            break;
        }
    }

    if (flag)
        cout << str.substr(1) << endl;
    else
        cout << "NO" << endl;

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


### STL 简化模拟

> [!NOTE] **[LeetCode 2122. 还原原数组](https://leetcode-cn.com/problems/recover-the-original-array/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 原以为枚举会比较麻烦，实际上使用 STL 可以很便捷...
> 
> 反复练习

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    vector<int> recoverArray(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        // nums[0] 必为 lower 的第一个元素，接下来枚举 higher 的第一个元素
        for (int i = 1; i < nums.size(); ++ i ) {
            int k = nums[i] - nums[0];
            if (k == 0 || k % 2)
                continue;
            
            // STL 大幅简化以下模拟过程
            multiset<int> H(nums.begin(), nums.end());
            vector<int> res;
            while (H.size()) {
                // x: lower 的下一个元素
                int x = *H.begin(); H.erase(H.begin());
                // x + k: higher 的下一个元素
                auto it = H.find(x + k);
                if (it == H.end())
                    break;
                H.erase(it);
                res.push_back(x + k / 2);
            }
            if (res.size() == nums.size() / 2)
                return res;
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

### 数字相关

> [!NOTE] **[LeetCode 7. 整数反转](https://leetcode-cn.com/problems/reverse-integer/)**
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
    int reverse(int x) {
        int res = 0;
        while (x) {
            int v = x % 10; x /= 10;
            if (res > INT_MAX / 10 || res == INT_MAX / 10 && v > 7) return 0;
            if (res < INT_MIN / 10 || res == INT_MIN / 10 && v < -8) return 0;
            res = res * 10 + v;
        }
        return res;
    }
};
```

##### **Python**

```python
# 推荐以下写法，直接按照每一位来进行反转，依次把 x 的低位放到 res 的高位

class Solution:
    def reverse(self, x: int) -> int:
        sign = 1  # 用sign标记这是个负数 还是个正数，可以省好几行代码
        if x < 0 : sign = -1
        x = x * sign

        n = 0
        while x :
            n = n * 10 + x % 10
            x //= 10

        return n * sign if n < 2 ** 31 else 0
      
# 这种方法太复杂了...非常不推荐
class Solution:
    def reverse(self, x: int) -> int:
        s = str(x)
        s_list = list(s)
        s = s_list
        n = len(s)
        flag = False
        if s[0] == '-':
            flag = True
            i = 1 
        else:i = 0
        j = n - 1 
        while i < j:
            s[i], s[j] = s[j], s[i]
            i += 1 
            j -= 1
        if flag:
            tmp = ''.join(s[1:])
            s = str(s[0]) + tmp.lstrip('0')
        else:
            tmp = ''.join(s)
            s = tmp[:-1].lstrip('0') + tmp[-1]
        if int(s) > (1 << 31) - 1 or int(s) < - ((1 << 31) - 1):return 0
        return int(s)

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 8. 字符串转换整数 (atoi)](https://leetcode-cn.com/problems/string-to-integer-atoi/)**
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
    int myAtoi(string s) {
        int n = s.size();
        int p = 0, f = 1;
        while (p < n && s[p] == ' ') ++ p ;
        if (s[p] == '+') ++ p ;
        else if (s[p] == '-') f = -1, ++ p ;
        int res = 0;
        while (p < n && isdigit(s[p])) {
            int v = s[p] - '0';
            if (res > INT_MAX / 10 || res == INT_MAX / 10 && v > 7) return INT_MAX;
            if (res < INT_MIN / 10 || res == INT_MIN / 10 && v > 8) return INT_MIN; // ATTENTION > 8
            res = res * 10 + f * v;
            ++ p ;
        }
        return res;
    }
};
```

##### **Python**

```python
# # python3
# 该学习的地方：用sign来区分 正负数
# 用一个 sign 来表示 这是一个 正数 【sign = 1】还是 负数 【sign = -1】 
# 1. 先去掉前面的空格
# 2. 再判断 空格 后的第一个字符是不是‘+‘ / ’-‘ ，如果是负数的话，要把 sign = -1
# 3. 开始循环，只有当 s[i] 在【0，9】之间才会加入到结果中。用num=0【非法情况 也是返回0，所以可以初始化位0】
# 4. 计算完成后要根据 sign 判断当前数 是 正数 还是 负数；最后再判断是否超出范围。返回结果即可。

class Solution:
    def strToInt(self, s: str) -> int:
        i, sign = 0, 1
        while i < len(s) and s[i] == ' ':i += 1

        if i < len(s) and s[i] == '+':
            i += 1
        elif i < len(s) and s[i] == '-':
            i += 1 
            sign = -1
        
        num = 0
        while i < len(s) and '0' <= s[i] <= '9':
            num = num * 10 + int(s[i])
            i += 1
        

        num *= sign
        maxv = (1 << 31)

        if num > 0 and num > maxv - 1:
            num = maxv - 1 
        if num < 0 and num < -maxv:
            num = -maxv
        return num
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 12. 整数转罗马数字](https://leetcode-cn.com/problems/integer-to-roman/)**
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
    string intToRoman(int num) {
        string res;
        int nums[]{1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1};
        string romans[]{"M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"};
        int p = 0;
        while (p < 13) {
            while (num >= nums[p]) num -= nums[p], res += romans[p];
            ++p;
        }
        return res;
    }
};
```

##### **Python**

```python
# 使用字典，记录每个数值 对应的 字符，并且是把大数值写在前面
# 对于 num, 遍历字典的 key, 看当前数是否大于这个key

class Solution:
    def intToRoman(self, num: int) -> str:
        my_dict ={1000: 'M', 900:'CM', 500: 'D', 400: 'CD', 100: 'C', 90: 'XC', 50: 'L', 40: 'XL', 10: 'X', 9: 'IX', 5: 'V', 4: 'IV', 1: 'I' }  # 使用哈希表，按照从大到小顺序排列
        # dic = OrderedDict({1000: 'M', 900: 'CM', 500: 'D', 400: 'CD', 100: 'C', 90: 'XC', 50: 'L', 40: 'XL', 10: 'X', 9: 'IX', 5: 'V', 4: 'IV', 1: 'I'})  #用这个也可以！直接排序！
        res = ''
        for key in my_dict:
            if num // key != 0:
                cnt = num // key 
                res += my_dict[key] * cnt 
                # print('res: {0}'.format(res))
                num %= key 
        return res
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 13. 罗马数字转整数](https://leetcode-cn.com/problems/roman-to-integer/)**
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
    int getValue(char c) {
        if (c == 'I') return 1;
        else if (c == 'V') return 5;
        else if (c == 'X') return 10;
        else if (c == 'L') return 50;
        else if (c == 'C') return 100;
        else if (c == 'D') return 500;
        else return 1000;
        //else if (c == 'M') return 1000;
    }
    int romanToInt(string s) {
        int len = s.size();
        int res = 0, tl, tr;
        for (int i = 0; i < len-1; ++ i ) {
            tl = getValue(s[i]);
            tr = getValue(s[i + 1]);
            if (tl < tr) res -= tl;
            else res += tl;
        }
        res += getValue(s[len-1]);
        return res;
    }
};
```

##### **Python**

```python
# 定义映射，将单一字母映射到数字。
# 从前往后扫描，如果发现 s[i+1] 的数字比 s[i] 的数字大，res减去当前数；否则直接累计 s[i] 的值。
# 最后一个数字一定是加上去的，所以第一个if判断里 要加一个条件 i + 1 < n


# 更好理解的写法：最后一个数单独拎出来，因为最后一个数字总是要加上去的
class Solution:
    def romanToInt(self, s: str) -> int:
        my_dict = {'I':1, 'V': 5, 'X' :10, 'L' :50, 'C' : 100, 'D':500, 'M':1000}
        n = len(s)
        res = 0
        for i in range(n-1):
            if my_dict[s[i]] < my_dict[s[i+1]]:
                res -= my_dict[s[i]]
            else:
                res += my_dict[s[i]]
        res += my_dict[s[n-1]]
        return res


class Solution:
    def romanToInt(self, s: str) -> int:
        my_dict = {'I' : 1, 'V' : 5, 'X' : 10, 'L' : 50, 'C' : 100, 'D' : 500, 'M' : 1000}
        n = len(s)
        res = 0

        for i in range(n):
            if i+1 < n and my_dict[s[i]] < my_dict[s[i+1]]:  # 踩坑：i+1 < n这里要记得
                res -= my_dict[s[i]]
            else:
                res += my_dict[s[i]]
        return res
```

<!-- tabs:end -->
</details>

<br>

* * *

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

> [!NOTE] **[LeetCode 29. 两数相除](https://leetcode-cn.com/problems/divide-two-integers/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 如果不能用 LL 的 trick

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 1**

```cpp
class Solution {
public:
    int divide(int dividend, int divisor) {
        if (divisor == 1) return dividend;
        else if (divisor == -1) {
            if (dividend == INT_MIN) return INT_MAX;
            else return -dividend;
        }
        int sign = 1;
        // 或使用异或判断符号
        if ((dividend > 0 && divisor < 0) || (dividend < 0 && divisor > 0)) sign = -1;
        // 都转成负数 处理溢出
        if (dividend > 0) dividend = -dividend;
        if (divisor > 0) divisor = -divisor;
        if (dividend > divisor) return 0;
        int cnt = 0;
        int a = dividend;
        // 此时是负数
        while (a <= divisor) {
            int b = divisor, p = 1;
            while (b > INT_MIN - b && a <= b + b) {
                b += b;
                p += p;
            }
            cnt += p;
            a -= b;
        }
        if (sign == -1) cnt = -cnt;
        return cnt;
    }
};
```

##### **C++ 2**

```cpp
class Solution {
public:
    int divide(int x, int y) {
        typedef long long LL;
        vector<LL> exp;
        bool is_minus = false;
        if (x < 0 && y > 0 || x > 0 && y < 0) is_minus = true;

        LL a = abs((LL)x), b = abs((LL)y);
        for (LL i = b; i <= a; i = i + i) exp.push_back(i);

        LL res = 0;
        for (int i = exp.size() - 1; i >= 0; i -- )
            if (a >= exp[i]) {
                a -= exp[i];
                res += 1ll << i;
            }

        if (is_minus) res = -res;

        if (res > INT_MAX || res < INT_MIN) res = INT_MAX;

        return res;
    }
};
```

##### **Python**

```python
"""
这道题：除数能减去多少个被除数。
1. 采用倍增的思想
2. 先确定商的符号，然后把被除数和除数通通转为正数
3. 然后用被除数不停的减除数，直到小于除数的时候，用一个计数遍历记录总共减了多少次，即为商了。
"""
class Solution:
    def divide(self, dividend: int, divisor: int) -> int:
        sign = (dividend < 0) != (divisor < 0) 
        dividend, divisor = abs(dividend), abs(divisor)
        res = 0
        tmp = divisor
        i = 1
        while dividend >= divisor:
            i = 1
            tmp = divisor
            while dividend >= tmp:
                dividend -= tmp
                tmp += tmp
                res += i
                # 代表有几个 dividend 
                i += i   
           
"""
           while divd >= tmp:
                divd -= tmp
                res += i
                i <<= 1
                tmp <<= 1
"""

        return min(2**31-1, max(-res if sign else res, -2**31))
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 43. 字符串相乘](https://leetcode-cn.com/problems/multiply-strings/)**
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
    string multiply(string num1, string num2) {
        vector<int> A, B;
        int n = num1.size(), m = num2.size();
        for (int i = n - 1; i >= 0; i -- ) A.push_back(num1[i] - '0');
        for (int i = m - 1; i >= 0; i -- ) B.push_back(num2[i] - '0');

        vector<int> C(n + m);
        for (int i = 0; i < n; i ++ )
            for (int j = 0; j < m; j ++ )
                C[i + j] += A[i] * B[j];

        for (int i = 0, t = 0; i < C.size(); i ++ ) {
            t += C[i];
            C[i] = t % 10;
            t /= 10;
        }

        int k = C.size() - 1;
        while (k > 0 && !C[k]) k -- ;

        string res;
        while (k >= 0) res += C[k -- ] + '0';

        return res;
    }
};
```

##### **Python**

```python
# python
"""
高精度乘法：先不考虑进位， 最后再从后向前遍历进位。
"""
class Solution:
    def multiply(self, num1: str, num2: str) -> str:
        s1, s2 = num1[::-1], num2[::-1]
        n, m = len(num1), len(num2)
        c = [0] * (n+m)
        
        # 先遍历所有位 
        for i in range(n): 
            for j in range(m):
                c[i+j] += int(s1[i]) * int(s2[j])
        
        t = 0
        # 再统一考虑进位
        for i in range(len(c)):
            t += c[i]
            c[i] = t % 10
            t //= 10 
        
        # 用字符串存储答案，方便去除掉高位0
        res = ''  
        for i in range(len(c)-1, -1, -1):
            res += str(c[i])
        # 去掉高位0，以及防止最后一个为0
        res = res[:-1].lstrip('0') + res[-1] 
        return res
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 65. 有效数字](https://leetcode-cn.com/problems/valid-number/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 标准写法

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
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
        // scanUnsignedInteger scanInteger 必须放前面
        // 避免短路原则导致 i 没有 ++ 
        if (i < n && s[i] == '.')
            flag = scanUnsignedInt(s, ++ i) || flag ;
        if (i < n && (s[i] == 'e' || s[i] == 'E'))
            flag = scanInt(s, ++ i) && flag;
        
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

> [!NOTE] **[LeetCode 66. 加一](https://leetcode-cn.com/problems/plus-one/)**
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
    vector<int> plusOne(vector<int>& digits) {
        reverse(digits.begin(), digits.end());
        int t = 1;
        for (auto & x: digits) {
            t += x;
            x = t % 10;
            t /= 10;
        }
        if (t) digits.push_back(t);

        reverse(digits.begin(), digits.end());

        return digits;
    }
};
```

##### **Python**

```python
"""
模拟人工加法的过程。
1. 从低位到高位，依次计算出每一位数字，过程中需要记录进位。
2. 如果最高位进位是1，则需要将整个数组后移一位，并在第0个位置写上1。
【时间复杂度分析：整个数组只遍历了一遍，所以时间复杂度是 O(n)。】
"""
class Solution:
    def plusOne(self, nums: List[int]) -> List[int]:
        n = len(nums)
        nums[n-1] += 1
        res, t, i = [], 0,  n - 1
        while i >= 0 or t:
            if i >= 0:
                t = t + nums[i]
            res.append(t % 10)
            t //= 10 
            i -= 1
        return res[::-1]
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 165. 比较版本号](https://leetcode-cn.com/problems/compare-version-numbers/)**
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
    int compareVersion(string s1, string s2) {
        int i = 0, j = 0;
        while (i < s1.size() || j < s2.size()) {
            int x = i, y = j;
            while (x < s1.size() && s1[x] != '.') x ++ ;
            while (y < s2.size() && s2[y] != '.') y ++ ;
            int a = (i == x) ? 0 : stoi(s1.substr(i, x - i));
            int b = (j == y) ? 0 : stoi(s2.substr(j, y - j));
            if (a > b) return 1;
            else if (a < b) return -1;
            i = x + 1; j = y + 1;
        }
        return 0;
    }
};
```

##### **Python**

```python
# 借助split分割方法
class Solution:
    def compareVersion(self, s1: str, s2: str) -> int:
        v1, v2 = s1.split('.'), s2.split('.')
        for i in range(max(len(v1), len(v2))):
            d1 = int(v1[i]) if i < len(v1) else 0
            d2 = int(v2[i]) if i < len(v2) else 0
            if d1 > d2:
                return 1
            elif d1 < d2:
                return -1
        return 0
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 166. 分数到小数](https://leetcode-cn.com/problems/fraction-to-recurring-decimal/)**
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
    string fractionToDecimal(int numerator, int denominator) {
        LL x = numerator, y = denominator;
        if (x % y == 0) return to_string(x / y);
        
        string res;
        if ((x < 0) ^ (y < 0)) res += '-';
        x = abs(x), y = abs(y);
        res += to_string(x / y) + '.', x %= y;

        unordered_map<LL, int> hash;    // 记录下标
        while (x) {
            hash[x] = res.size();
            x *= 10;
            res += to_string(x / y);
            x %= y;
            if(hash.count(x)) {
                res = res.substr(0, hash[x]) + '(' + res.substr(hash[x]) + ')';
                break;
            }
        }
        return res;
    }
};
```

##### **Python**

```python
"""
为了方便处理，我们先将所有负数运算转化为正数运算。
然后再算出分数的整数部分，再将精力集中在小数部分。

计算小数部分的难点在于如何判断是否是循环小数，以及找出循环节的位置。
回忆手工计算除法的过程，每次将余数乘10再除以除数，当同一个余数出现两次时，我们就找到了循环节。
所以我们可以用一个哈希表 unordered_map<int,int> 记录所有余数所对应的商在小数点后第几位，当计算到相同的余数时，上一次余数的位置和当前位置之间的数，就是该小数的循环节。

时间复杂度分析：计算量与结果长度成正比，是线性的。所以时间复杂度是 O(n)。

"""
class Solution:
    def fractionToDecimal(self, x: int, y: int) -> str:
        res = ''
        if x % y == 0:
            return str(x // y)
        if (x < 0) ^ (y < 0): # 异号
            x, y = abs(x), abs(y)
            res += '-'
        res += str(x // y) + '.'
        x = x % y

        seen_dict = {}
        while x:
            seen_dict[x] = len(res)
            x *= 10
            res += str(x // y)
            x = x % y
            if x in seen_dict:
                res = res[:seen_dict[x]] + '(' +  res[seen_dict[x]:] + ')'
                break
        return res
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 179. 最大数](https://leetcode-cn.com/problems/largest-number/)**
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
// yxc
class Solution {
public:
    string largestNumber(vector<int>& nums) {
        sort(nums.begin(), nums.end(), [](int x, int y) {
            string a = to_string(x), b = to_string(y);
            return a + b > b + a;
        });
        string res;
        for (auto x: nums) res += to_string(x);
        int k = 0;
        while (k + 1 < res.size() && res[k] == '0') k ++ ;
        return res.substr(k);
    }
};
```

##### **Python**

```python
#1. 先把nums中的所有数字转化为字符串，形成字符串数组 nums_str
#2. 比较两个字符串x,y的拼接结果x+y和y+x哪个更大，从而确定x和y谁排在前面；将nums_str降序排序
#3. 把整个数组排序的结果拼接成一个字符串，并且返回

import functools
class Solution:
    def largestNumber(self, nums: List[int]) -> str:
        nums_str = list(map(str,nums))
        compare = lambda x, y: 1 if x + y < y + x else -1
        nums_str.sort(key = functools.cmp_to_key(compare))
        res = ''.join(nums_str)
        if res[0] == '0':  # 判断最终拼接完的字符串中首位是不是 "0"，因为如果 nums 至少有一个数字不是 0， 那该数字一定会排在所有的 0 的前面
            res = '0'
        return res

      
      # 区别在于cmp函数的写法（熟悉熟悉）
class Solution:
    def largestNumber(self, nums):
        from functools import cmp_to_key
        temp = list(map(str,nums))
        temp.sort(key = cmp_to_key(lambda x,y:int(x+y)-int(y+x)),reverse = True )
        return ''.join(temp if temp[0]!='0' else '0')

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 273. 整数转换英文表示](https://leetcode-cn.com/problems/integer-to-english-words/)**
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
    string num0_19[20] = {
        "Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven",
        "Eight", "Nine", "Ten", "Eleven", "Twelve", "Thirteen",
        "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen",
        "Nineteen",
    };
    string num20_90[8] = {
        "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy",
        "Eighty", "Ninety",
    };
    string num1000[4] = {
        "Billion ", "Million ", "Thousand ", "",
    };

    // 返回 1 ~ 999 的英文表示
    string get(int x) {
        string res;
        if (x >= 100) {
            res += num0_19[x / 100] + " Hundred ";
            x %= 100;
        }
        if (x >= 20) {
            res += num20_90[x / 10 - 2] + " ";
            x %= 10;
            if (x) res += num0_19[x] + ' ';
        } else if (x) res += num0_19[x] + ' ';
        return res;
    }

    string numberToWords(int num) {
        if (!num) return "Zero";
        string res;
        for (int i = 1e9, j = 0; i >= 1; i /= 1000, ++ j )
            if (num >= i) {
                res += get(num / i) + num1000[j];
                num %= i;
            }
        res.pop_back();
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

> [!NOTE] **[LeetCode 405. 数字转换为十六进制数](https://leetcode-cn.com/problems/convert-a-number-to-hexadecimal/)**
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
    string toHex(int num) {
        unsigned int unum = num;
        if (!unum) return "0";
        string res, nums = "0123456789abcdef";
        while (unum) {
            res += nums[unum & 0xf];
            unum >>= 4;
        }
        reverse(res.begin(), res.end());
        return res;
    }

    string toHex_2(int num) {
        if (!num) return "0";
        string res;
        for (int i = 0; i < 8; ++ i ) {
            int v = num & 15;
            //cout << v << endl;
            if (!num) break;
            if (v < 10) res.push_back('0' + v);
            else res.push_back('a' + v - 10);
            num >>= 4;
        }
        reverse(res.begin(), res.end());
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

### 螺旋矩阵

> [!NOTE] **[LeetCode 54. 螺旋矩阵](https://leetcode-cn.com/problems/spiral-matrix/)**
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
    vector<int> spiralOrder(vector<vector<int>>& matrix) {
        vector<int> res;
        if (matrix.empty()) return res;
        int m = matrix.size(), n = matrix[0].size();
        int u = 0, d = m - 1, l = 0, r = n - 1;
        for (;;) {
            for (int i = l; i <= r; ++ i ) res.push_back(matrix[u][i]);
            if ( ++ u > d) break;
            for (int i = u; i <= d; ++ i ) res.push_back(matrix[i][r]);
            if ( -- r < l) break;
            for (int i = r; i >= l; -- i ) res.push_back(matrix[d][i]);
            if ( -- d < u) break;
            for (int i = d; i >= u; -- i ) res.push_back(matrix[i][l]);
            if ( ++ l > r) break;
        }
        return res;
    }
};
```

##### **Python**

```python
class Solution:
    def spiralOrder(self, a: List[List[int]]) -> List[int]:
        n, m = len(a), len(a[0])
        L, R, T, B = 0, m-1, 0, n-1
        res = []
        while True:
            for i in range(L, R + 1):res.append(a[T][i])
            T += 1
            if T > B:break 
            for i in range(T, B + 1):res.append(a[i][R])
            R -= 1
            if L > R:break 
            for i in range(R, L - 1, -1):res.append(a[B][i])   # 踩坑：细节 L - 1
            B -= 1
            if T > B:break
            for i in range(B, T - 1, -1):res.append(a[i][L])
            L += 1
            if L > R:break 
        return res
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 59. 螺旋矩阵 II](https://leetcode-cn.com/problems/spiral-matrix-ii/)**
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
    vector<vector<int>> generateMatrix(int n) {
        vector<vector<int>> res(n, vector<int>(n));
        int u = 0, d = n - 1, l = 0, r = n - 1;
        int p = 0;
        for (;;) {
            for (int i = l; i <= r; ++ i ) res[u][i] = ++ p ;
            if ( ++ u > d) break;
            for (int i = u; i <= d; ++ i ) res[i][r] = ++ p ;
            if ( -- r < l) break;
            for (int i = r; i >= l; -- i ) res[d][i] = ++ p ;
            if ( -- d < u) break;
            for (int i = d; i >= u; -- i ) res[i][l] = ++ p ;
            if ( ++ l > r) break;
        }
        return res;
    }
};
```

##### **Python**

```python
# 和上一道题完全一样
class Solution:
    def generateMatrix(self, n: int) -> List[List[int]]:
        L, R, T, B = 0, n-1, 0, n-1
        p = 0 
        res = [[0] * n for _ in range(n)]
        while True:
            for i in range(L, R + 1):
                p += 1
                res[T][i] = p
            T += 1 
            if T > B:break 
            for i in range(T, B + 1):
                p += 1
                res[i][R] = p 
            R -= 1
            if L > R:break 
            for i in range(R, L - 1, -1):
                p += 1
                res[B][i] = p
            B -= 1
            if T > B:break 
            for i in range(B, T - 1, -1):
                p += 1
                res[i][L] = p
            L += 1
            if L > R:break 
        return res 
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 6111. 螺旋矩阵 IV](https://leetcode.cn/problems/spiral-matrix-iv/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 加了个链表

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    vector<vector<int>> g;
    vector<vector<int>> spiralMatrix(int m, int n, ListNode* head) {
        this->g = vector<vector<int>>(m, vector<int>(n, -1));
        int u = 0, d = m - 1, l = 0, r = n - 1;
        for (;;) {
            for (int i = l; i <= r && head; ++ i )
                g[u][i] = head->val, head = head->next;
            if (head == nullptr || ++ u > d)
                break;
            for (int i = u; i <= d && head; ++ i )
                g[i][r] = head->val, head = head->next;
            if (head == nullptr || -- r < l)
                break;
            for (int i = r; i >= l && head; -- i )
                g[d][i] = head->val, head = head->next;
            if (head == nullptr || -- d < u)
                break;
            for (int i = d; i >= u && head; -- i )
                g[i][l] = head->val, head = head->next;
            if (head == nullptr || ++ l > r)
                break;
        }
        return g;
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

> [!NOTE] **[LeetCode 1914. 循环轮转矩阵](https://leetcode-cn.com/problems/cyclically-rotating-a-grid/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 模拟即可

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    vector<vector<int>> g;
    int n, m;
    
    vector<vector<int>> rotateGrid(vector<vector<int>>& grid, int k) {
        this->g = grid;
        this->n = grid.size(), m = grid[0].size();
        
        int l = 0, r = m - 1, u = 0, d = n - 1;
        vector<vector<int>> all;
        for (;;) {
            vector<int> t;
            for (int i = l; i <= r; ++ i )
                t.push_back(g[u][i]);
            if ( ++ u > d) {
                all.push_back(t);
                break;
            }
            for (int i = u; i <= d; ++ i )
                t.push_back(g[i][r]);
            if ( -- r < l) {
                all.push_back(t);
                break;
            }
            for (int i = r; i >= l; -- i )
                t.push_back(g[d][i]);
            if ( -- d < u) {
                all.push_back(t);
                break;
            }
            for (int i = d; i >= u; -- i )
                t.push_back(g[i][l]);
            if ( ++ l > r) {
                all.push_back(t);
                break;
            }
            all.push_back(t);
        }
        
        vector<vector<int>> res = g;
        int p = 0, asz = all.size();
        l = 0, r = m - 1, u = 0, d = n - 1;
        while (p < asz) {
            auto t = all[p ++ ];
            int sz = t.size(), id = 0;
            int tk = k;
            tk %= sz;
            
            for (int i = l; i <= r; ++ i , ++ id )
                res[u][i] = t[(id + tk) % sz];
            if ( ++ u > d)
                break;
            
            for (int i = u; i <= d; ++ i , ++ id )
                res[i][r] = t[(id + tk) % sz];
            if ( -- r < l)
                break;
            
            for (int i = r; i >= l; -- i , ++ id )
                res[d][i] = t[(id + tk) % sz];
            if ( -- d < u)
                break;
            
            for (int i = d; i >= u; -- i , ++ id )
                res[i][l] = t[(id + tk) % sz];
            if ( ++ l > r)
                break;
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

### excel 表

> [!NOTE] **[LeetCode 168. Excel表列名称](https://leetcode-cn.com/problems/excel-sheet-column-title/)**
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
    string convertToTitle(int n) {
        int k = 1;
        for (long long p = 26; n > p; p *= 26) {
            n -= p;
            k ++ ;
        }

        n -- ;
        string res;
        while (k -- ) {
            res += n % 26 + 'A';
            n /= 26;
        }

        reverse(res.begin(), res.end());
        return res;
    }
};
```

##### **C++**

```cpp
class Solution {
public:
    string convertToTitle(int n) {
        string res;
        while (n) {
            int mod = (n - 1) % 26;
            res.push_back('A' + mod);
            n = (n - 1) / 26;
        }
        reverse(res.begin(), res.end());
        return res;
    }
};
```

##### **Python**

```python
# 进制转换题 ： 【除留余数法】 ==> 参照 一个数字如何从10进制 转化到 10进制的。
#  每一次循环里，先取 %，得到当前位的（第一次是个位）数；然后进入下一次循环前， 取 // 
#  10 进制包括数字：0~9； 16 进制包括：0-15；  26 进制应包括：0~25
#  因为 Excel 取值范围为 1~26，故可将 26 进制 逻辑上的 个位、十位、百位…均减 1 映射到 0~25 即可，最后转换为字符。

class Solution:
    def convertToTitle(self, n: int) -> str:
        s = ''
        while n:
            n -= 1
            s = chr(n % 26 + 65) + s
            n = n // 26
        return s
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 171. Excel表列序号](https://leetcode-cn.com/problems/excel-sheet-column-number/)**
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
    int titleToNumber(string s) {
        int a = 0;
        for (long long i = 0, p = 26; i < s.size() - 1; i ++, p *= 26)
            a += p;

        int b = 0;
        for (auto c: s) b = b * 26 + c - 'A';
        return a + b + 1;
    }
};
```

##### **C++ 2**

```cpp
class Solution {
public:
    int titleToNumber(string s) {
        int res = 0;
        for (auto & c : s) {
            int v = c - 'A' + 1;
            res = res * 26 + v;
        }
        return res;
    }
};
```

##### **Python**

```python
# 进制转换，类比其他进制怎么转换的。同理，需要注意的是 这里A 代表 1.（而不是0）
class Solution:
    def titleToNumber(self, s: str) -> int:
        res = 0
        for x in s:
            res *= 26
            res += ord(x) - 65 + 1
        return res
```

<!-- tabs:end -->
</details>

<br>

* * *