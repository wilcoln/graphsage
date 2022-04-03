python experiments/table1/run.py --batch_size=512 --num_epochs=10 --ignore_datasets reddit citeseer
python experiments/fig2a/run.py --batch_size=512 --num_epochs=10 --dataset=citation --no-show
python experiments/fig2b/run.py --batch_size=512 --num_epochs=10 --dataset=citation --num_runs=5 --no-show
python experiments/fig3/run.py --batch_size=512 --num_epochs=10 --dataset=citation --no-show