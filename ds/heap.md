

堆是一棵树，其每个节点都有一个键值，且每个节点的键值都大于等于/小于等于其父亲的键值。

每个节点的键值都大于等于其父亲键值的堆叫做小根堆，否则叫做大根堆。[STL 中的 `priority_queue`](/lang/csl/container-adapter/#_13) 其实就是一个大根堆。

（小根）堆主要支持的操作有：插入一个数、查询最小值、删除最小值、合并两个堆、减小一个元素的值。

一些功能强大的堆（可并堆）还能（高效地）支持 merge 等操作。

一些功能更强大的堆还支持可持久化，也就是对任意历史版本进行查询或者操作，产生新的版本。

## 堆的分类

|         操作\\数据结构        |                                  配对堆                                  |      二叉堆     |      左偏树     |      二项堆     |    斐波那契堆    |
| :---------------------: | :-------------------------------------------------------------------: | :----------: | :----------: | :----------: | :---------: |
|        插入（insert）       |                                 $O(1)$                                |  $O(\log n)$ |  $O(\log n)$ |    $O(1)$    |    $O(1)$   |
|     查询最小值（find-min）     |                                 $O(1)$                                |    $O(1)$    |    $O(1)$    |  $O(\log n)$ |    $O(1)$   |
|    删除最小值（delete-min）    |                              $O(\log n)$                              |  $O(\log n)$ |  $O(\log n)$ |  $O(\log n)$ | $O(\log n)$ |
|        合并 (merge)       |                                 $O(1)$                                |    $O(n)$    |  $O(\log n)$ |  $O(\log n)$ |    $O(1)$   |
| 减小一个元素的值 (decrease-key) | $o(\log n)$（下界 $\Omega(\log \log n)$，上界 $O(2^{2\sqrt{\log \log n}})$) |  $O(\log n)$ |  $O(\log n)$ |  $O(\log n)$ |    $O(1)$   |
|         是否支持可持久化        |                                $\times$                               | $\checkmark$ | $\checkmark$ | $\checkmark$ |   $\times$  |

习惯上，不加限定提到“堆”时往往都指二叉堆。

> [!NOTE] **[AcWing 839. 模拟堆](https://www.acwing.com/problem/content/841/)**
> 
> 题意: TODO

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 100010;

int h[N], ph[N], hp[N], cnt;

void heap_swap(int a, int b) {
    swap(ph[hp[a]], ph[hp[b]]);
    swap(hp[a], hp[b]);
    swap(h[a], h[b]);
}

void down(int u) {
    int t = u;
    if (u * 2 <= cnt && h[u * 2] < h[t]) t = u * 2;
    if (u * 2 + 1 <= cnt && h[u * 2 + 1] < h[t]) t = u * 2 + 1;
    if (u != t) {
        heap_swap(u, t);
        down(t);
    }
}

void up(int u) {
    while (u / 2 && h[u] < h[u / 2]) {
        heap_swap(u, u / 2);
        u >>= 1;
    }
}

int main() {
    int n, m = 0;
    cin >> n;
    while (n -- ) {
        char op[5];
        int k, x;
        cin >> op;
        if (op[0] == 'I') {
            cin >> x;
            cnt ++ ;
            m ++ ;
            ph[m] = cnt, hp[cnt] = m;
            h[cnt] = x;
            up(cnt);
        } else if (op[0] == 'P' && op[1] == 'M') cout << h[1] << endl;
        else if (op[0] == 'D' && op[1] == 'M') {
            heap_swap(1, cnt);
            cnt -- ;
            down(1);
        } else if (op[0] == 'D') {
            cin >> k;
            k = ph[k];
            heap_swap(k, cnt);
            cnt -- ;
            up(k);
            down(k);
        } else {
            cin >> k >> x;
            k = ph[k];
            h[k] = x;
            up(k);
            down(k);
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

## 习题

### heap

> [!NOTE] **[LeetCode 373. 查找和最小的K对数字](https://leetcode.cn/problems/find-k-pairs-with-smallest-sums/)**
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
    typedef tuple<int, int, int> tpi3;
    vector<vector<int>> kSmallestPairs(vector<int>& nums1, vector<int>& nums2, int k) {
        if (nums1.empty() || nums2.empty()) return {};
        int n = nums1.size(), m = nums2.size();
        priority_queue<tpi3, vector<tpi3>, greater<tpi3>> heap;
        for (int i = 0; i < m; ++ i )
            heap.push({nums2[i] + nums1[0], 0, i});
        vector<vector<int>> res;
        while (k -- && heap.size()) {
            auto [v, p1, p2] = heap.top(); heap.pop();
            res.push_back({nums1[p1], nums2[p2]});
            if (p1 + 1 < n)
                heap.push({nums1[p1 + 1] + nums2[p2], p1 + 1, p2});
        }
        return res;
    }
};

// tuple 也可以直接用 vector
// yxc
class Solution {
public:
    typedef vector<int> VI;
    vector<vector<int>> kSmallestPairs(vector<int>& a, vector<int>& b, int k) {
        if (a.empty() || b.empty()) return {};
        int n = a.size(), m = b.size();
        priority_queue<VI, vector<VI>, greater<VI>> heap;
        for (int i = 0; i < m; i ++ ) heap.push({b[i] + a[0], 0, i});
        vector<VI> res;
        while (k -- && heap.size()) {
            auto t = heap.top();
            heap.pop();
            res.push_back({a[t[1]], b[t[2]]});
            if (t[1] + 1 < n)
                heap.push({a[t[1] + 1] + b[t[2]], t[1] + 1, t[2]});
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

> [!NOTE] **[LeetCode 658. 找到 K 个最接近的元素](https://leetcode.cn/problems/find-k-closest-elements/)**
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
    vector<int> findClosestElements(vector<int>& arr, int k, int x) {
        priority_queue<pair<int, int>> heap;
        for (auto v: arr) {
            heap.push({abs(x - v), v});
            if (heap.size() > k) heap.pop();
        }
        vector<int> res;
        while (heap.size()) res.push_back(heap.top().second), heap.pop();
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

> [!NOTE] **[LeetCode 692. 前K个高频单词](https://leetcode.cn/problems/top-k-frequent-words/)**
> 
> 题意: 尝试以 $O(n log k)$ 时间复杂度和 $O(n)$ 空间复杂度解决。

> [!TIP] **思路**
> 
> 原地建堆 复杂度要求

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using PIS = pair<int, string>;
    
    vector<string> topKFrequent(vector<string>& words, int k) {
        unordered_map<string, int> cnt;
        for (auto& w: words) cnt[w] ++ ;

        vector<PIS> ws;
        for (auto [k, v]: cnt) ws.push_back({v, k});
        auto cmp = [](PIS a, PIS b) {
            if (a.first != b.first) return a.first < b.first;
            return a.second > b.second;
        };
        make_heap(ws.begin(), ws.end(), cmp);
        
        vector<string> res;
        while (k -- ) {
            res.push_back(ws[0].second);
            pop_heap(ws.begin(), ws.end(), cmp);
            ws.pop_back();
        }

        return res;
    }
};
```

##### **Python**

```python
# 1. 用哈希表统计每个单词出现的次数； 2. 原地建堆，取前k个；
class Solution:
    def topKFrequent(self, words: List[str], k: int) -> List[str]:
        # my_cnt = collections.Counter(words) 也可以直接用计数器 进行计数。 
        my_dict = collections.defaultdict(int)
        for c in words:
            my_dict[c] += 1 
        
        q = []
        for key, val in my_dict.items():
            heapq.heappush(q, (-val, key))  #  堆的元素可以是元组/列表 类型; 小根堆 所以 取负数
        res = []
        for _ in range(k):
            res.append(heapq.heappop(q)[1])
        return res
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 1439. 有序矩阵中的第 k 个最小数组和](https://leetcode.cn/problems/find-the-kth-smallest-sum-of-a-matrix-with-sorted-rows/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 任意一行选一个数 求第k小
> 
> 解决办法：
> 
> - 二分数组和 求个数 个数比k个大就缩小右边界 ==> 并不友好
> 
> - 维护小顶堆 每次取出一个组合就把这个后面可拓展的组合全部加入堆 使用 set 对组合去重 ==> **正解**（参考算法竞赛进阶指南做法

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
struct dwell {
    vector<int> pos;
    int sum;
    dwell(const vector<int>& _pos, int _sum) : pos(_pos), sum(_sum) {}
    bool operator<(const dwell& that) const { return sum > that.sum; }
};

class Solution {
public:
    int kthSmallest(vector<vector<int>>& mat, int k) {
        int m = mat.size(), n = mat[0].size();

        auto getsum = [&](const vector<int>& v) {
            int ret = 0;
            for (int i = 0; i < m; ++i) { ret += mat[i][v[i]]; }
            return ret;
        };

        set<vector<int>> seen;

        vector<int> init(m, 0);
        seen.insert(init);
        priority_queue<dwell> q;
        q.emplace(init, getsum(init));
        for (int _ = 0; _ < k - 1; ++_) {
            auto [pos, sum] = q.top();
            q.pop();
            for (int i = 0; i < m; ++i) {
                if (pos[i] + 1 < n) {
                    ++pos[i];
                    if (!seen.count(pos)) {
                        q.emplace(pos, getsum(pos));
                        seen.insert(pos);
                    }
                    --pos[i];
                }
            }
        }

        auto fin = q.top();
        return fin.sum;
    }
};
// zerotrac2
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 1801. 积压订单中的订单总数](https://leetcode.cn/problems/number-of-orders-in-the-backlog/)**
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
    const int MOD = 1e9 + 7;
    using PII = pair<int, int>;
    using LL = long long;
    int getNumberOfBacklogOrders(vector<vector<int>>& orders) {
        priority_queue<PII, vector<PII>, greater<PII>> sells;
        priority_queue<PII, vector<PII>, less<PII>> buys;
        int n = orders.size();
        for (int i = 0; i < n; ++ i ) {
            auto & os = orders[i];
            int cnt = os[1], type = os[2], price = os[0];
            if (type) {
                while (buys.size() && buys.top().first >= price) {
                    auto [t_price, t_cnt] = buys.top(); buys.pop();
                    int cost = min(cnt, t_cnt);
                    cnt -= cost, t_cnt -= cost;
                    if (t_cnt) {
                        buys.push({t_price, t_cnt});
                        break;
                    }
                }
                // 1 sell
                if (cnt)
                    sells.push({price, cnt});
            } else {
                while (sells.size() && sells.top().first <= price) {
                    auto [t_price, t_cnt] = sells.top(); sells.pop();
                    int cost = min(cnt, t_cnt);
                    cnt -= cost, t_cnt -= cost;
                    if (t_cnt) {
                        sells.push({t_price, t_cnt});
                        break;
                    }
                }
                // 0 buy
                if (cnt)
                    buys.push({price, cnt});
            }
        }
        
        LL res = 0;
        while (buys.size()) {
            auto [p, c] = buys.top(); buys.pop();
            res = (res + c) % MOD;
        }
        while (sells.size()) {
            auto [p, c] = sells.top(); sells.pop();
            res = (res + c) % MOD;
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

> [!NOTE] **[LeetCode 1847. 最近的房间](https://leetcode.cn/problems/closest-room/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 显然堆维护即可
> 
> 最初写的时候 `lower_bound(S.begin(), S.end(), pid)` TLE
> 
> 注意 STL set 使用 `S.lower_bound(pid)`

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using PII = pair<int, int>;
    using TIII = tuple<int, int, int>;
    
    vector<int> closestRoom(vector<vector<int>>& rooms, vector<vector<int>>& queries) {
        vector<PII> ve;
        for (auto & r : rooms)
            ve.push_back({r[1], r[0]});
        sort(ve.begin(), ve.end());
        reverse(ve.begin(), ve.end());
        
        vector<TIII> qs;
        int m = queries.size();
        for (int i = 0; i < m; ++ i )
            qs.push_back({queries[i][1], queries[i][0], i});
        sort(qs.begin(), qs.end());
        // 反向 面积从大到小 条件越来越严格
        reverse(qs.begin(), qs.end());
        
        vector<int> res(m, -1);
        int p = 0, n = ve.size();
        set<int> S;
        for (auto & [msz, pid, id] : qs) {
            while (p < n && ve[p].first >= msz) {
                S.insert(ve[p].second);
                p ++ ;
            }
            if (S.size()) {
                auto it = S.lower_bound(pid); // ATTENTION lower_bound(S.begin(), S.end(), pid); 就会超时
                int t = 2e9;
                if (it != S.end()) {
                    if (abs(*it - pid) < abs(t - pid))
                        t = *it;
                }
                if (it != S.begin()) {
                    it -- ;
                    if (abs(*it - pid) <= abs(t - pid))
                        t = *it;
                }
                res[id] = t;
            }
        }
        return res;
    }
};
```

##### **C++ trick**

```cpp
class Solution {
public:
    // using PII = pair<int, int>;
    // using TIII = tuple<int, int, int>;
    
    vector<int> closestRoom(vector<vector<int>>& rooms, vector<vector<int>>& queries) {
        auto cmp = [&](auto & a, auto & b){
            return a[1] > b[1];
        };
        sort(rooms.begin(), rooms.end(), cmp);
        int m = queries.size();
        for (int i = 0; i < m; ++ i )
            queries[i].push_back(i);
        sort(queries.begin(), queries.end(), cmp);
        
        vector<int> res(m, -1);
        int p = 0, n = rooms.size();
        set<int> S;
        for (auto & q : queries) {
            int pid = q[0], msz = q[1], id = q[2];
            while (p < n && rooms[p][1] >= msz) {
                S.insert(rooms[p][0]);
                p ++ ;
            }
            if (S.size()) {
                auto it = S.lower_bound(pid);
                int t = 2e9;
                if (it != S.end()) {
                    if (*it - pid < t - pid)
                        t = *it;
                }
                if (it != S.begin()) {
                    it -- ;
                    if (pid - *it <= t - pid)
                        t = *it;
                }
                res[id] = t;
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

### 双堆模拟 一般用于占用类问题

> [!NOTE] **[LeetCode 1606. 找到处理最多请求的服务器](https://leetcode.cn/problems/find-servers-that-handled-most-number-of-requests/)** [TAG]
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

> [!NOTE] **[LeetCode 1882. 使用服务器处理任务](https://leetcode.cn/problems/process-tasks-using-servers/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 显然堆
> 
> 一开始WA: 问题在于同一时刻开始多个任务的实现细节

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ WA**

```cpp
// WA 30 / 34
class Solution {
public:
    using PII = pair<int, int>;
    
    vector<int> assignTasks(vector<int>& servers, vector<int>& tasks) {
        int n = servers.size(), m = tasks.size();
        vector<int> res(m, -1);
        
        priority_queue<PII, vector<PII>, greater<PII>> idle, working;
        priority_queue<int, vector<int>, greater<int>> task;
        
        for (int i = 0; i < n; ++ i )
            idle.push({servers[i], i});
        
        for (int i = 0; i < m; ++ i ) {
            // 当前遍历到第i个任务 时间也为i 弹出所有到i时刻已完成的任务服务器
            while (working.size()) {
                auto [et, sid] = working.top();
                if (et > i)
                    break;
                else {
                    working.pop();
                    idle.push({servers[sid], sid});
                }
            }
            // 加入当前第i个任务
            task.push(i);
            // 无可用服务器 continue
            if (idle.empty())
                continue;
            
            // 取下标最小的任务
            auto tid = task.top(); task.pop();
            int t = tasks[tid];

            // 存在可用服务器 即使用最优的服务器
            auto [rank, sid] = idle.top(); idle.pop();
            res[tid] = sid;
            working.push({i + t, sid});
        }
        
        // now时间戳
        int now = m;
        while (task.size()) {
            // 根据时间戳更新服务器状态
            while (working.size()) {
                auto [et, sid] = working.top();
                if (et > now)
                    break;
                else {
                    working.pop();
                    idle.push({servers[sid], sid});
                }
            }
            
            // 取下标最小的任务
            auto tid = task.top();
            int t = tasks[tid];
            
            // 如果有可用服务器 则使用该服务器
            // 否则更新时间戳 快进到会有服务器被弹出的时刻
            if (idle.size()) {
                task.pop();
                auto [rank, sid] = idle.top(); idle.pop();
                res[tid] = sid;
                working.push({now + t, sid});
            } else {
                auto [et, sid] = working.top();
                now = et;
            }
        }
        return res;
    }
};
```

##### **C++ AC**

```cpp
class Solution {
public:
    using PII = pair<int, int>;
    
    vector<int> assignTasks(vector<int>& servers, vector<int>& tasks) {
        int n = servers.size(), m = tasks.size();
        vector<int> res(m, -1);
        
        priority_queue<PII, vector<PII>, greater<PII>> idle, working;
        priority_queue<int, vector<int>, greater<int>> task;
        
        for (int i = 0; i < n; ++ i )
            idle.push({servers[i], i});
        
        for (int i = 0; i < m; ++ i ) {
            // 当前遍历到第i个任务 时间也为i 弹出所有到i时刻已完成的任务服务器
            while (working.size()) {
                auto [et, sid] = working.top();
                if (et > i)
                    break;
                else {
                    working.pop();
                    idle.push({servers[sid], sid});
                }
            }
            // 加入当前第i个任务
            task.push(i);
            // 无可用服务器 continue
            if (idle.empty())
                continue;
            
            // ===== 问题在于 此时此刻可能有多个任务可以同时开始 =====
            // 取下标最小的任务
            while (task.size() && idle.size()) {
                auto tid = task.top(); task.pop();
                int t = tasks[tid];

                // 存在可用服务器 即使用最优的服务器
                auto [rank, sid] = idle.top(); idle.pop();
                res[tid] = sid;
                working.push({i + t, sid});
            }
        }
        
        // now时间戳
        int now = m;
        while (task.size()) {
            // 根据时间戳更新服务器状态
            while (working.size()) {
                auto [et, sid] = working.top();
                if (et > now)
                    break;
                else {
                    working.pop();
                    idle.push({servers[sid], sid});
                }
            }
            
            // 取下标最小的任务
            auto tid = task.top();
            int t = tasks[tid];
            
            // 如果有可用服务器 则使用该服务器
            // 否则更新时间戳 快进到会有服务器被弹出的时刻
            if (idle.size()) {
                task.pop();
                auto [rank, sid] = idle.top(); idle.pop();
                res[tid] = sid;
                working.push({now + t, sid});
            } else {
                auto [et, sid] = working.top();
                now = et;
            }
        }
        return res;
    }
};
```

##### **C++ 标准**

```cpp
class Solution {
public:
    using PII = pair<int, int>;
    
    vector<int> assignTasks(vector<int>& servers, vector<int>& tasks) {
        int n = servers.size(), m = tasks.size();
        
        priority_queue<PII, vector<PII>, greater<PII>> busy, idle;
        
        for (int i = 0; i < n; ++ i )
            idle.push({servers[i], i});
        
        int ts = 0;
        vector<int> res(m);
        for (int i = 0; i < m; ++ i ) {
            ts = max(ts, i);
            while (busy.size() && busy.top().first <= ts) {
                auto [_, sid] = busy.top(); busy.pop();
                idle.push({servers[sid], sid});
            }
            
            if (idle.empty()) {
                ts = busy.top().first;
                i = i - 1;  // 更新时间戳 再次检查本个任务
            } else {
                auto [rank, sid] = idle.top(); idle.pop();
                res[i] = sid;
                busy.push({ts + tasks[i], sid});
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

> [!NOTE] **[LeetCode 1942. 最小未被占据椅子的编号](https://leetcode.cn/problems/the-number-of-the-smallest-unoccupied-chair/)**
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

> [!NOTE] **[LeetCode 2402. 会议室 III](https://leetcode.cn/problems/meeting-rooms-iii/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 标准做法是双堆模拟

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 暴力**

```cpp
class Solution {
public:
    using LL = long long;
    const static int N = 110;
    
    LL last[N], cnt[N]; // start end 比较大 要用 LL
    
    void init() {
        memset(last, 0, sizeof last);
        memset(cnt, 0, sizeof cnt);
    }
    
    int mostBooked(int n, vector<vector<int>>& meetings) {
        init();
        sort(meetings.begin(), meetings.end());
        
        for (auto & m : meetings) {
            int l = m[0], r = m[1];
            int p = -1;
            for (int i = 0; i < n; ++ i )
                if (last[i] <= l) {
                    p = i;
                    break;
                }
            if (p == -1) {
                for (int i = 0; i < n; ++ i )
                    if (p == -1 || last[i] < last[p])
                        p = i;
                last[p] += r - l;
            } else {
                last[p] = r;
            }
            cnt[p] ++ ;
        }
        
        int p = -1;
        for (int i = 0; i < n; ++ i )
            if (p == -1 || cnt[i] > cnt[p])
                p = i;
        return p;
    }
};
```

##### **C++ 双堆模拟**

```cpp
class Solution {
public:
    using LL = long long;
    using PLI = pair<LL, int>;
    const static int N = 110;

    int cnt[N];

    int mostBooked(int n, vector<vector<int>>& meetings) {
        sort(meetings.begin(), meetings.end());
        priority_queue<int, vector<int>, greater<>> idle;
        priority_queue<PLI, vector<PLI>, greater<>> working;
        
        for (int i = 0; i < n; ++ i )
            idle.push(i);
        
        for (auto & m : meetings) {
            LL l = m[0], r = m[1], p = -1;
            while (!working.empty() && working.top().first <= l)
                idle.push(working.top().second), working.pop();
            if (idle.empty()) {
                auto [e, i] = working.top(); working.pop();
                r = e + r - l;
                p = i;
            } else {
                p = idle.top(); idle.pop();
            }
            cnt[p] ++ ;
            working.push({r, p});
        }
        int p = -1;
        for (int i = 0; i < n; ++ i )
            if (p == -1 || cnt[i] > cnt[p])
                p = i;
        return p;
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

> [!NOTE] **[LeetCode 2532. 过桥的时间](https://leetcode.cn/problems/time-to-cross-a-bridge/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
>
> 工人共有四种状态，每一种状态都存在优先级对比，大致可以分为两类：
>
> 1. 在左右对岸等待的工人，优先级根据题目中给定进行计算
> 
> 2. 在搬起或者放下的工人，优先级根据完成时间来计算
>
> 因此，要设置 **<u>四个优先队列</u>** 来分别存储他们。**当旧仓库还有货物或者右边还有人时，过程就需要继续**：
>
> - 如果搬起或者放下的工人在此时已经完成，则将他们依次加入左边或者右边的等待队列
> 
> - 如果右边有人等待，则取优先级最低的工人过桥，过桥后放入左侧的处于「放下」状态的队列
> 
> - 否则，如果旧仓库还有货物，并且左侧有等待的工人，则取优先级最低的工人过桥，过桥后放入右侧处于「搬起」状态的队列，并使得旧仓库待搬运货物数量减一
> 
> - 否则，此时没有人需要过桥，时间应该过渡到第一个处于「放下」或者「搬起」状态的工人切换状态的时刻【**相当于时间戳快进**】【思考为什么可以】

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 左侧：关注放下箱子后的时间，对于初始化情况都为 0
    // 右侧：关注拿到箱子后的时间，后侧初始化空
    // ==> 尝试推导答案所在的范围：最差情况下 1e4 * 4000 = 4e7
    // 
    // 还需要记录桥使用的状态 => last
    // 【ATTENTION】还需要记录已经有足够的人运送 n 个箱子（左侧的不需要再往右走）
    // 如果仅考虑 {time, index} 二维偏序 ==> wrong
    // 【ATTENTION】需要同时考虑效率，在满足时间的前提下找效率最低的（之前没有考虑这一个纬度）
    
    using PII = pair<int, int>;
    
    int findCrossingTime(int n, int k, vector<vector<int>>& time) {
        // 大顶堆 按照效率降序
        // {leftToRight + rightToLeft, i}
        priority_queue<PII> idle_l, idle_r;
        // 小顶堆 按时间戳升序
        // {timestamp, i}
        priority_queue<PII, vector<PII>, greater<PII>> l, r;
        
        // init
        for (int i = 0; i < k; ++ i )
            idle_l.push({time[i][0] + time[i][2], i});
        
        int t = 0;
        // ATTENTION: 条件
        while (n || r.size() || idle_r.size()) {
            while (l.size() && l.top().first <= t) {
                int i = l.top().second; l.pop();
                idle_l.push({time[i][0] + time[i][2], i});  // 按效率加入
            }
            while (r.size() && r.top().first <= t) {
                int i = r.top().second; r.pop();
                idle_r.push({time[i][0] + time[i][2], i});  // 按效率加入
            }
            
            if (idle_r.size()) {
                int i = idle_r.top().second; idle_r.pop();
                t += time[i][2];
                l.push({t + time[i][3], i});    // 按时间加入
            } else if (idle_l.size() && n) {
                n -- ;
                int i = idle_l.top().second; idle_l.pop();
                t += time[i][0];
                r.push({t + time[i][1], i});
            } else {
                // ATTENTION 时间戳快进
                t = 1e9;
                if (l.size())
                    t = min(t, l.top().first);
                if (r.size())
                    t = min(t, r.top().first);
            }
        }
        
        return t;
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

### multiset

> [!NOTE] **[LeetCode 218. 天际线问题](https://leetcode.cn/problems/the-skyline-problem/)**
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
    vector<vector<int>> getSkyline(vector<vector<int>>& buildings) {
        vector<pair<int, int>> h;
        multiset<int> m;
        vector<vector<int>> res;
        for (auto &b : buildings) {
            h.push_back({b[0], -b[2]});
            h.push_back({b[1], b[2]});
        }
        sort(h.begin(), h.end());
        m.insert(0);
        int prev = 0;
        for (auto i : h) {
            if (i.second < 0)
                m.insert(-i.second);
            else
                m.erase(m.find(i.second));
            int cur = *m.rbegin();
            if (cur != prev) {
                res.push_back({i.first, cur});
                prev = cur;
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

> [!NOTE] **[LeetCode 480. 滑动窗口中位数](https://leetcode.cn/problems/sliding-window-median/)**
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
    int k;
    // r.size() >= l.size()
    multiset<int> left, right;

    double get_medium() {
        if (k % 2) return *right.begin();
        // 这里写double 处理整数溢出
        return ((double)*left.rbegin() + *right.begin()) / 2;
    }
    vector<double> medianSlidingWindow(vector<int>& nums, int _k) {
        k = _k;
        for (int i = 0; i < k; ++ i ) right.insert(nums[i]);
        // 较小的一半压入left
        for (int i = 0; i < k / 2; ++ i ) {
            left.insert(*right.begin());
            right.erase(right.begin());
        }
        vector<double> res;
        res.push_back(get_medium());

        for (int i = k; i < nums.size(); ++ i ) {
            int x = nums[i], y = nums[i - k];

            if (x >= *right.begin()) right.insert(x);
            else left.insert(x);
            if (y >= *right.begin()) right.erase(right.find(y));
            else left.erase(left.find(y));

            while (left.size() > right.size()) {
                right.insert(*left.rbegin());
                // rbegin 必须写find
                left.erase(left.find(*left.rbegin()));
            }
            while (left.size() + 1 < right.size()) {
                left.insert(*right.begin());
                right.erase(right.begin());
            }
            res.push_back(get_medium());
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

### 理论推导 -> 简化堆模拟

> [!NOTE] **[LeetCode 3266. K 次乘运算后的最终数组 II](https://leetcode.cn/problems/final-array-state-after-k-multiplication-operations-ii/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 如果模拟 操作次数可能达到 1e9
> 
> 考虑结合其特性，在完成部分模拟操作后直接 break 以简化计算
> 
> 【套路题】

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // After the k operations, apply modulo 109 + 7 to every value in nums.
    // => 在此之前 堆内元素都是 LL
    //
    // 【套路题】
    // 首先用最小堆手动模拟操作，直到原数组的最大值 mx 成为这 n 个数的最小值
    // 随后不需要手动模拟 假设此时还剩下k次操作 则
    //  - 对于前 k mod n 小的数字 还可以操作 k/n + 1 次
    //  - 其余元素 操作 k/n 次
    
    using LL = long long;
    using PLI = pair<LL, int>;
    const static int MOD = 1e9 + 7;
    
    LL qpow(LL a, LL b) {
        LL ret = 1;
        while (b) {
            if (b & 1)
                ret = ret * a % MOD;
            a = a * a % MOD;
            b >>= 1;    // ATTENTION
        }
        return ret;
    }
    
    vector<int> getFinalState(vector<int>& nums, int k, int multiplier) {
        if (multiplier == 1)
            return nums;
        
        int n = nums.size();
        
        // greater<>()
        priority_queue<PLI, vector<PLI>, greater<>> heap;
        for (int i = 0; i < n; ++ i )
            heap.push({nums[i], i});
        
        LL maxv = ranges::max(nums);    // ATTENTION trick
        // ATTENTION: k 消耗完 || 堆顶是 maxv
        for ( ; k && heap.top().first < maxv; -- k ) {
            auto [v, idx] = heap.top(); heap.pop(); // 不能 &
            heap.push({v * multiplier, idx});
        }
        
        // 剩余的直接用公式算
        vector<PLI> xs;
        while (heap.size())
            xs.push_back(heap.top()), heap.pop();
        sort(xs.begin(), xs.end());
        
        vector<int> res(n);
        for (int i = 0; i < n; ++ i ) {
            auto & [v, idx] = xs[i];
            res[idx] = v % MOD * qpow(multiplier, k / n + (i < k % n)) % MOD;
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