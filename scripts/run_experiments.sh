python experiments/table1/run.py --batch_size=512 --num_epochs=10 --ignore_datasets reddit citeseer --no_extensions
python experiments/fig2a/run.py --batch_size=512 --num_epochs=10 --dataset=reddit --no_extensions --no_show
python experiments/fig2b/run.py --batch_size=512 --num_epochs=10 --dataset=citation --num_runs=5 --no_show
python experiments/fig3/run.py --batch_size=512 --num_epochs=10 --dataset=citation --no_extensions --no_show