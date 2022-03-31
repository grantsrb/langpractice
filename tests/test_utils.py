from langpractice.utils.utils import *
import unittest

class TestUtils(unittest.TestCase):
    def test_zipfian1(self):
        n_loops = 5000
        low = 1
        high = 10
        order = 1
        counts = {i:0 for i in range(low, high+1)}
        tot = 0
        for i in range(n_loops):
            samp = zipfian(low, high, order)
            counts[samp] += 1
            tot += 1
        targ_probs = {k:1/(k**order) for k in counts.keys()}
        s = np.sum(list(targ_probs.values()))
        targ_probs = {k:v/s for k,v in targ_probs.items()}
        for k,v in counts.items():
            prob = v/tot
            diff = prob-targ_probs[k]
            self.assertTrue(np.abs(diff) < 0.03)

    def test_zipfian2(self):
        n_loops = 5000
        low = 1
        high = 10
        order = 2
        counts = {i:0 for i in range(low, high+1)}
        tot = 0
        for i in range(n_loops):
            samp = zipfian(low, high, order)
            counts[samp] += 1
            tot += 1
        targ_probs = {k:1/(k**order) for k in counts.keys()}
        s = np.sum(list(targ_probs.values()))
        targ_probs = {k:v/s for k,v in targ_probs.items()}
        for k,v in counts.items():
            prob = v/tot
            diff = prob-targ_probs[k]
            self.assertTrue(np.abs(diff) < 0.03)

    def test_piraha_labels(self):
        weights = {
            3:   torch.FloatTensor([.55, .45]),
            4:   torch.FloatTensor([.4, .6]),
            5:   torch.FloatTensor([.4, .6]),
            6:   torch.FloatTensor([.4, .6]),
            7:   torch.FloatTensor([.45, .55]),
            8:   torch.FloatTensor([.3, .7]),
            9:   torch.FloatTensor([.3, .7]),
            10:  torch.FloatTensor([.3, .7]),
        }
        n_items = torch.randint(0,11, (100,))
        avgs = torch.zeros_like(n_items)
        n_loops = 5000
        for i in range(n_loops):
            labels = torch.zeros_like(n_items)
            labels = get_piraha_labels(labels,n_items)
            avgs = avgs + labels
        avgs = avgs/n_loops
        for k in weights.keys():
            targ = weights[k][0]*2 + weights[k][1]*3
            avg = avgs[n_items==k]
            if len(avg) > 0:
                avg = avg.mean()
                diff = float(avg-targ)
                self.assertTrue(np.abs(diff)<0.01)

    def test_duplicate_labels(self):
        n_items = torch.randint(0,11, (100,))
        avgs = torch.zeros_like(n_items)
        n_loops = 5000
        for i in range(n_loops):
            labels = torch.zeros_like(n_items)
            labels = get_duplicate_labels(labels,n_items, 22)
            if i < 20:
                for n,l in zip(n_items,labels):
                    self.assertTrue((n*2)==l or (n*2+1)==l)
            avgs = avgs + labels
        avgs = avgs/n_loops
        for i in range(torch.max(n_items)):
            avg = avgs[n_items==i]
            if len(avg) > 0:
                avg = avg.mean()
                targ = ((i*2)+.5)
                diff = targ-avg
                self.assertTrue(np.abs(diff)<0.01)

    def test_get_lang_labels_english(self):
        max_label = 10
        use_count_words = 1
        n_samps = 100
        n_items = torch.randint(0,max_label+10, (n_samps,))
        n_targs = n_items.clone()
        labels = get_lang_labels(
            n_items,
            n_targs,
            max_label,
            use_count_words
        )
        labels = labels.cpu().detach().numpy()
        n_items = n_items.cpu().detach().numpy()
        idx = n_items<max_label
        self.assertTrue(np.array_equal(labels[idx],n_items[idx]))
        self.assertTrue(
            np.array_equal(
                labels[~idx],
                np.ones_like(labels[~idx])*max_label
            )
        )

    def test_get_lang_labels_comparison(self):
        max_label = 10
        use_count_words = 0
        n_samps = 100
        n_items = torch.randint(0,max_label+10, (n_samps,))
        n_targs = torch.randint(0,max_label+10, (n_samps,))
        labels = get_lang_labels(
            n_items,
            n_targs,
            max_label,
            use_count_words
        )
        labels = labels.cpu().detach().numpy()
        n_items = n_items.cpu().detach().numpy()
        n_targs = n_targs.cpu().detach().numpy()
        idx = n_items<n_targs
        goal = np.zeros_like(labels[idx])
        self.assertTrue(np.array_equal(labels[idx],goal))
        idx = n_items==n_targs
        goal = np.ones_like(labels[idx])
        self.assertTrue(np.array_equal(labels[idx],goal))
        idx = n_items>n_targs
        goal = np.ones_like(labels[idx])*2
        self.assertTrue(np.array_equal(labels[idx],goal))

    def test_calc_accs(self):
        logits = torch.FloatTensor([[
            [1,2,3,4],
            [4,1,2,3],
            [-1,2,-3,-4],
            [-100,-5,100,0],
        ]])
        # 0 correct
        targs = torch.LongTensor([[ 0, 1, 0, 0, ]])
        accs = calc_accs(logits, targs,prepender="test")
        self.assertEqual(0, accs["test_acc"])
        # 1 correct
        targs = torch.LongTensor([[ 0, 0, 0, 0, ]])
        accs = calc_accs(logits, targs,prepender="test")
        self.assertEqual(1/4, accs["test_acc"])
        # all correct
        targs = torch.LongTensor([[ 3,0,1,2 ]])
        accs = calc_accs(logits, targs,prepender="test")
        self.assertEqual(1, accs["test_acc"])

    def test_calc_accs_categories(self):
        logits = torch.FloatTensor([
            [
                [1,2,3,4],
                [4,1,2,3],
                [-1,2,-3,-4],
                [-100,-5,100,0],
            ],
            [
                [1,2,3,4],
                [4,1,2,3],
                [-1,2,-3,-4],
                [-100,-5,100,0],
            ],
        ])
        categories = torch.LongTensor([
            [ 1, 1, 3, 0, ],
            [ 0, 3, 0, 3, ],
        ])
        # 0 correct
        targs = torch.LongTensor([
            [ 0, 1, 0, 0, ],
            [ 0, 1, 0, 0, ]
        ])
        accs = calc_accs(logits, targs, categories,prepender="test")
        self.assertEqual(0, accs["test_acc_0"])
        self.assertEqual(0, accs["test_acc_1"])
        self.assertEqual(0, accs["test_acc_3"])
        # 1 correct 1
        targs = torch.LongTensor([
            [ 0, 0, 0, 0, ],
            [ 0, 1, 0, 0, ]
        ])
        accs = calc_accs(logits, targs, categories,prepender="test")
        self.assertEqual(0, accs["test_acc_0"])
        self.assertEqual(1/2, accs["test_acc_1"])
        self.assertEqual(0, accs["test_acc_3"])
        # all correct 0
        targs = torch.LongTensor([
            [ 0, 0, 0, 2, ],
            [ 3, 1, 1, 0, ]
        ])
        accs = calc_accs(logits, targs, categories,prepender="test")
        self.assertEqual(1, accs["test_acc_0"])
        self.assertEqual(1/2, accs["test_acc_1"])
        self.assertEqual(0, accs["test_acc_3"])

    def test_avg_over_dicts(self):
        vals = np.arange(10)
        dicts = [ {"foo": i, "poo": i*i} for i in vals ]
        avgs = avg_over_dicts(dicts)
        self.assertEqual(np.mean(vals), avgs["foo"])
        self.assertEqual(np.mean(vals**2), avgs["poo"])

    def test_calc_lang_loss_and_accs(self):
        loss_fxn = torch.nn.CrossEntropyLoss()
        langs = (torch.FloatTensor([
            [
                [1,2,3,4],
                [4,1,2,3],
                [-1,2,-3,-4],
                [-100,-5,100,0],
            ],
            [
                [1,2,3,4],
                [4,1,2,3],
                [-1,2,-3,-4],
                [-100,-5,100,0],
            ]
        ]).cuda(),)
        drops = torch.LongTensor([
            [ 1,1,1,1 ],
            [ 1,1,1,1 ],
        ])
        # 0 correct
        targs = torch.LongTensor([
            [ 0, 1, 0, 0, ],
            [ 0, 1, 0, 0, ],
        ])
        targ_loss = loss_fxn(langs[0].reshape(-1,4), targs.cuda().reshape(-1))
        targ_accs = calc_accs(langs[0].cpu(), targs, targs, prepender="test_lang")
        loss, accs = calc_lang_loss_and_accs(
            langs,
            targs.reshape(-1),
            drops.reshape(-1),
            loss_fxn,
            prepender="test"
        )
        self.assertEqual(float(loss), float(targ_loss))
        for k in targ_accs.keys():
            self.assertEqual(targ_accs[k], accs[k])
        # 3 correct
        targs = torch.LongTensor([
            [ 3, 1, 1, 0, ],
            [ 0, 0, 0, 0, ],
        ])
        targ_loss = loss_fxn(langs[0].reshape(-1,4), targs.cuda().reshape(-1))
        targ_accs = calc_accs(langs[0].cpu(), targs, targs, prepender="test_lang")
        loss, accs = calc_lang_loss_and_accs(
            langs,
            targs.reshape(-1),
            drops.reshape(-1),
            loss_fxn,
            prepender="test"
        )
        self.assertEqual(float(loss), float(targ_loss))
        for k in targ_accs.keys():
            self.assertEqual(targ_accs[k], accs[k])

    def test_calc_lang_loss_and_accs_drops(self):
        loss_fxn = torch.nn.CrossEntropyLoss()
        langs = (torch.FloatTensor([
            [
                [1,2,3,4],
                [4,1,2,3],
                [-1,2,-3,-4],
                [-100,-5,100,0],
            ],
            [
                [1,2,3,4],
                [4,1,2,3],
                [-1,2,-3,-4],
                [-100,-5,100,0],
            ]
        ]).cuda(),)
        drops = torch.LongTensor([
            [ 0,1,0,1 ],
            [ 1,0,1,0 ],
        ])
        dropped_lang = torch.FloatTensor([
            [
                [4,1,2,3],
                [-100,-5,100,0],
            ],
            [
                [1,2,3,4],
                [-1,2,-3,-4],
            ]
        ]).cuda()
        # 0 correct
        targs = torch.LongTensor([
            [ 0, 1, 0, 0, ],
            [ 0, 1, 0, 0, ],
        ])
        dropped_targs = torch.LongTensor([
            [ 1, 0, ],
            [ 0, 0, ],
        ])
        targ_loss = loss_fxn(
            dropped_lang.reshape(-1,4),
            dropped_targs.cuda().reshape(-1)
        )
        targ_accs = calc_accs(
            dropped_lang.cpu(),
            dropped_targs,
            dropped_targs,
            prepender="test_lang"
        )
        loss, accs = calc_lang_loss_and_accs(
            langs,
            targs.reshape(-1),
            drops.reshape(-1),
            loss_fxn,
            prepender="test"
        )
        self.assertEqual(float(loss), float(targ_loss))
        for k in targ_accs.keys():
            self.assertEqual(targ_accs[k], accs[k])
        # 3 correct
        targs = torch.LongTensor([
            [ 3, 0, 1, 2, ],
            [ 0, 0, 0, 0, ],
        ])
        dropped_targs = torch.LongTensor([
            [ 0, 2, ],
            [ 0, 0, ],
        ])
        targ_loss = loss_fxn(
            dropped_lang.reshape(-1,4),
            dropped_targs.cuda().reshape(-1)
        )
        targ_accs = calc_accs(
            dropped_lang.cpu(),
            dropped_targs,
            dropped_targs,
            prepender="test_lang"
        )
        loss, accs = calc_lang_loss_and_accs(
            langs,
            targs.reshape(-1),
            drops.reshape(-1),
            loss_fxn,
            prepender="test"
        )
        self.assertEqual(float(loss), float(targ_loss))
        for k in targ_accs.keys():
            self.assertEqual(targ_accs[k], accs[k])

    def test_calc_actn_loss_and_accs(self):
        loss_fxn = torch.nn.CrossEntropyLoss()
        actns = torch.FloatTensor([
            [
                [1,2,3,4],
                [4,1,2,3],
                [-1,2,-3,-4],
                [-100,-5,100,0],
            ],
            [
                [1,2,3,4],
                [4,1,2,3],
                [-1,2,-3,-4],
                [-100,-5,100,0],
            ]
        ]).cuda()
        n_targs = torch.LongTensor([
            [ 0, 1, 2, 3 ],
            [ 4, 3, 2, 1 ],
        ])
        # 0 correct
        targs = torch.LongTensor([
            [ 0, 1, 0, 0, ],
            [ 0, 1, 0, 0, ],
        ])
        targ_loss = loss_fxn(actns.reshape(-1,4), targs.cuda().reshape(-1))
        targ_accs = calc_accs(
            actns.cpu(),
            targs,
            n_targs,
            prepender="test_actn"
        )
        loss, accs = calc_actn_loss_and_accs(
            actns,
            targs.reshape(-1),
            n_targs.reshape(-1),
            loss_fxn,
            prepender="test"
        )
        self.assertEqual(float(loss), float(targ_loss))
        for k in targ_accs.keys():
            self.assertEqual(targ_accs[k], accs[k])
        # 3 correct
        targs = torch.LongTensor([
            [ 3, 1, 1, 0, ],
            [ 0, 0, 0, 0, ],
        ])
        targ_loss = loss_fxn(actns.reshape(-1,4), targs.cuda().reshape(-1))
        targ_accs = calc_accs(
            actns.cpu().reshape(-1,4),
            targs.reshape(-1),
            n_targs.reshape(-1),
            prepender="test_actn"
        )
        loss, accs = calc_actn_loss_and_accs(
            actns,
            targs.reshape(-1),
            n_targs.reshape(-1),
            loss_fxn,
            prepender="test"
        )
        self.assertEqual(float(loss), float(targ_loss))
        for k in targ_accs.keys():
            self.assertEqual(targ_accs[k], accs[k])

    def test_get_loss_and_accs_phase0(self):
        phase = 0
        loss_fxn = torch.nn.CrossEntropyLoss()
        preds = (torch.FloatTensor([
            [
                [1,2,3,4],
                [4,1,2,3],
                [-1,2,-3,-4],
                [-100,-5,100,0],
            ],
            [
                [1,2,3,4],
                [4,1,2,3],
                [-1,2,-3,-4],
                [-100,-5,100,0],
            ]
        ]).cuda(),)
        drops = torch.LongTensor([
            [ 0,1,0,1 ],
            [ 1,0,1,0 ],
        ])
        n_targs = torch.LongTensor([
            [ 0, 1, 2, 3 ],
            [ 4, 3, 2, 1 ],
        ])
        # 0 correct
        targs = torch.LongTensor([
            [ 0, 1, 0, 0, ],
            [ 0, 1, 0, 0, ],
        ])
        targ_loss, targ_accs = calc_lang_loss_and_accs(
            preds,
            targs.reshape(-1),
            drops.reshape(-1),
            loss_fxn,
            prepender="test"
        )
        loss, accs = get_loss_and_accs(
            phase,
            preds,
            preds,
            targs.reshape(-1),
            targs.reshape(-1),
            drops.reshape(-1),
            n_targs.reshape(-1),
            "test",
            loss_fxn,
            lang_p=0.5
        )

        self.assertEqual(float(loss), float(targ_loss))
        for k in targ_accs.keys():
            self.assertEqual(targ_accs[k], accs[k])
        # 3 correct
        targs = torch.LongTensor([
            [ 3, 1, 1, 0, ],
            [ 0, 0, 0, 0, ],
        ])
        targ_loss, targ_accs = calc_lang_loss_and_accs(
            preds,
            targs.reshape(-1),
            drops.reshape(-1),
            loss_fxn,
            prepender="test"
        )
        loss, accs = get_loss_and_accs(
            phase,
            preds,
            preds,
            targs.reshape(-1),
            targs.reshape(-1),
            drops.reshape(-1),
            n_targs.reshape(-1),
            "test",
            loss_fxn,
            lang_p=0.5
        )

        self.assertEqual(float(loss), float(targ_loss))
        for k in targ_accs.keys():
            self.assertEqual(targ_accs[k], accs[k])
    
    def test_get_loss_and_accs_phase1(self):
        phase = 1
        loss_fxn = torch.nn.CrossEntropyLoss()
        preds = torch.FloatTensor([
            [
                [1,2,3,4],
                [4,1,2,3],
                [-1,2,-3,-4],
                [-100,-5,100,0],
            ],
            [
                [1,2,3,4],
                [4,1,2,3],
                [-1,2,-3,-4],
                [-100,-5,100,0],
            ]
        ]).cuda()
        drops = torch.LongTensor([
            [ 0,1,0,1 ],
            [ 1,0,1,0 ],
        ])
        n_targs = torch.LongTensor([
            [ 0, 1, 2, 3 ],
            [ 4, 3, 2, 1 ],
        ])
        # 0 correct
        targs = torch.LongTensor([
            [ 0, 1, 0, 0, ],
            [ 0, 1, 0, 0, ],
        ])
        targ_loss, targ_accs = calc_actn_loss_and_accs(
            preds,
            targs.reshape(-1),
            n_targs.reshape(-1),
            loss_fxn,
            prepender="test"
        )
        loss, accs = get_loss_and_accs(
            phase,
            preds,
            preds,
            targs.reshape(-1),
            targs.reshape(-1),
            drops.reshape(-1),
            n_targs.reshape(-1),
            "test",
            loss_fxn,
            lang_p=0.5
        )

        self.assertEqual(float(loss), float(targ_loss))
        for k in targ_accs.keys():
            self.assertEqual(targ_accs[k], accs[k])
        # 3 correct
        targs = torch.LongTensor([
            [ 3, 1, 1, 0, ],
            [ 0, 0, 0, 0, ],
        ])
        targ_loss, targ_accs = calc_actn_loss_and_accs(
            preds,
            targs.reshape(-1),
            n_targs.reshape(-1),
            loss_fxn,
            prepender="test"
        )
        loss, accs = get_loss_and_accs(
            phase,
            preds,
            preds,
            targs.reshape(-1),
            targs.reshape(-1),
            drops.reshape(-1),
            n_targs.reshape(-1),
            "test",
            loss_fxn,
            lang_p=0.5
        )

        self.assertEqual(float(loss), float(targ_loss))
        for k in targ_accs.keys():
            self.assertEqual(targ_accs[k], accs[k])
    
    def test_get_loss_and_accs_phase2(self):
        phase = 2
        loss_fxn = torch.nn.CrossEntropyLoss()
        actns = torch.FloatTensor([
            [
                [1,2,3,4],
                [4,1,2,3],
                [-1,2,-3,-4],
                [-100,-5,100,0],
            ],
            [
                [1,2,3,4],
                [4,1,2,3],
                [-1,2,-3,-4],
                [-100,-5,100,0],
            ]
        ]).cuda()
        langs = (actns.clone(),)
        drops = torch.LongTensor([
            [ 0,1,0,1 ],
            [ 1,0,1,0 ],
        ])
        n_targs = torch.LongTensor([
            [ 0, 1, 2, 3 ],
            [ 4, 3, 2, 1 ],
        ])
        # 0 correct
        targs = torch.LongTensor([
            [ 0, 1, 0, 0, ],
            [ 0, 1, 0, 0, ],
        ])
        lang_targ_loss, lang_targ_accs = calc_lang_loss_and_accs(
            langs,
            targs.reshape(-1),
            drops.reshape(-1),
            loss_fxn,
            prepender="test"
        )
        actn_targ_loss, actn_targ_accs = calc_actn_loss_and_accs(
            actns,
            targs.reshape(-1),
            n_targs.reshape(-1),
            loss_fxn,
            prepender="test"
        )
        targ_loss = 0.5*lang_targ_loss + 0.5*actn_targ_loss
        targ_accs = {**lang_targ_accs, **actn_targ_accs}
        loss, accs = get_loss_and_accs(
            phase,
            actns,
            langs,
            targs.reshape(-1),
            targs.reshape(-1),
            drops.reshape(-1),
            n_targs.reshape(-1),
            "test",
            loss_fxn,
            lang_p=0.5
        )

        self.assertEqual(float(loss), float(targ_loss))
        for k in targ_accs.keys():
            self.assertEqual(targ_accs[k], accs[k])
        # 3 correct
        targs = torch.LongTensor([
            [ 3, 1, 1, 0, ],
            [ 0, 0, 0, 0, ],
        ])
        lang_targ_loss, lang_targ_accs = calc_lang_loss_and_accs(
            langs,
            targs.reshape(-1),
            drops.reshape(-1),
            loss_fxn,
            prepender="test"
        )
        actn_targ_loss, actn_targ_accs = calc_actn_loss_and_accs(
            actns,
            targs.reshape(-1),
            n_targs.reshape(-1),
            loss_fxn,
            prepender="test"
        )
        targ_loss = 0.5*lang_targ_loss + 0.5*actn_targ_loss
        targ_accs = {**lang_targ_accs, **actn_targ_accs}
        loss, accs = get_loss_and_accs(
            phase,
            actns,
            langs,
            targs.reshape(-1),
            targs.reshape(-1),
            drops.reshape(-1),
            n_targs.reshape(-1),
            "test",
            loss_fxn,
            lang_p=0.5
        )

        self.assertEqual(float(loss), float(targ_loss))
        for k in targ_accs.keys():
            self.assertEqual(targ_accs[k], accs[k])
    

if __name__=="__main__":
    unittest.main()
