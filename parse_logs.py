import re
import csv
import os


def parse_logs(filepath):
    train_rows = []
    val_rows = []
    test_rows = []

    with open(filepath, 'r') as f:
        lines = f.readlines()

    i = 0
    current_split = None
    current_epoch = None
    current_row = {}

    while i < len(lines):
        line = lines[i].strip()

        # Détecte le split et l'epoch
        m = re.match(r'^(Train|Val|Test) \[(\d+)\]$', line)
        if m:
            # Sauvegarde la ligne précédente si complète
            if current_row and current_split:
                if current_split == 'Train':
                    train_rows.append(current_row)
                elif current_split == 'Val':
                    val_rows.append(current_row)
                elif current_split == 'Test':
                    test_rows.append(current_row)

            current_split = m.group(1)
            current_epoch = int(m.group(2))
            current_row   = {'epoch': current_epoch}
            i += 1
            continue

        # Détecte les lignes de métriques
        m_loss = re.match(r'loss\s+\| g0: ([\d.]+) \| g1: ([\d.]+) \| g2: ([\d.]+) \| g3: ([\d.]+) \| avg: ([\d.]+) \| worst: ([\d.]+)', line)
        if m_loss:
            current_row['loss_g0'] = float(m_loss.group(1))
            current_row['loss_g1'] = float(m_loss.group(2))
            current_row['loss_g2'] = float(m_loss.group(3))
            current_row['loss_g3'] = float(m_loss.group(4))
            current_row['avg_loss'] = float(m_loss.group(5))
            current_row['worst_loss'] = float(m_loss.group(6))
            i += 1
            continue

        m_acc = re.match(r'accuracy \| g0: ([\d.]+) \| g1: ([\d.]+) \| g2: ([\d.]+) \| g3: ([\d.]+) \| avg: ([\d.]+) \| worst: ([\d.]+)', line)
        if m_acc:
            current_row['acc_g0'] = float(m_acc.group(1))
            current_row['acc_g1'] = float(m_acc.group(2))
            current_row['acc_g2'] = float(m_acc.group(3))
            current_row['acc_g3'] = float(m_acc.group(4))
            current_row['avg_acc'] = float(m_acc.group(5))
            current_row['worst_acc'] = float(m_acc.group(6))
            i += 1
            continue

        m_q = re.match(r'q\s+\| g0: ([\d.]+) \| g1: ([\d.]+) \| g2: ([\d.]+) \| g3: ([\d.]+)', line)
        if m_q:
            current_row['adv_prob_g0'] = float(m_q.group(1))
            current_row['adv_prob_g1'] = float(m_q.group(2))
            current_row['adv_prob_g2'] = float(m_q.group(3))
            current_row['adv_prob_g3'] = float(m_q.group(4))
            i += 1
            continue

        i += 1

    # Sauvegarde la dernière ligne
    if current_row and current_split:
        if current_split == 'Train':
            train_rows.append(current_row)
        elif current_split == 'Val':
            val_rows.append(current_row)
        elif current_split == 'Test':
            test_rows.append(current_row)

    return train_rows, val_rows, test_rows


def write_csv(rows, path, has_adv_probs=False):
    if not rows:
        return
    fieldnames = ['epoch', 'loss_g0', 'loss_g1', 'loss_g2', 'loss_g3', 'avg_loss', 'worst_loss',
                  'acc_g0', 'acc_g1', 'acc_g2', 'acc_g3', 'avg_acc', 'worst_acc']
    if has_adv_probs:
        fieldnames += ['adv_prob_g0', 'adv_prob_g1', 'adv_prob_g2', 'adv_prob_g3']

    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    print(f"Écrit : {path} ({len(rows)} lignes)")


def main(log_dir, i):
    train_rows, val_rows, test_rows = parse_logs(f'{log_dir}/logs_{i}.txt')

    train_last_row = train_rows[len(train_rows) - 1]
    train_weighted_avg_acc = (train_last_row['acc_g0'] * 3498 + train_last_row['acc_g1'] * 184 + train_last_row[
        'acc_g2'] * 56 +
                              train_last_row['acc_g3'] * 1057) / 4795

    val_last_row = val_rows[len(val_rows) - 1]
    val_weighted_avg_acc = (val_last_row['acc_g0'] * 467 + val_last_row['acc_g1'] * 466 + val_last_row['acc_g2'] * 133 +
                            val_last_row['acc_g3'] * 133) / 1199

    test_last_row = test_rows[len(test_rows) - 1]
    test_weighted_avg_acc = (test_last_row['acc_g0'] * 2255 + test_last_row['acc_g1'] * 2255 + test_last_row[
        'acc_g2'] * 642 +
                             test_last_row['acc_g3'] * 642) / 5794

    print(
        f'train_weighted_avg_acc : {train_weighted_avg_acc}\nval_weighted_avg_acc : {val_weighted_avg_acc}\ntest_weighted_avg_acc : {test_weighted_avg_acc}\n\n')

    train_avg_acc = train_last_row['avg_acc']
    val_avg_acc = val_last_row['avg_acc']
    test_avg_acc = test_last_row['avg_acc']

    print(f'train_avg_acc : {train_avg_acc}\nval_avg_acc : {val_avg_acc}\ntest_avg_acc : {test_avg_acc}\n\n')

    train_worst_acc = train_last_row['worst_acc']
    val_worst_acc = val_last_row['worst_acc']
    test_worst_acc = test_last_row['worst_acc']

    print(f'train_worst_acc : {train_worst_acc}\nval_worst_acc : {val_worst_acc}\ntest_worst_acc : {test_worst_acc}\n\n')

    write_csv(train_rows, f'{log_dir}/train.csv', has_adv_probs=True)
    write_csv(val_rows, f'{log_dir}/val.csv')
    write_csv(test_rows, f'{log_dir}/test.csv')

    print(f"\nTrain: {len(train_rows)} epochs")
    print(f"Val:   {len(val_rows)} epochs")
    print(f"Test:  {len(test_rows)} epochs")


if __name__ == '__main__':
    for i in range(5):
        main(f'logs/adjustments/{i}', i)
