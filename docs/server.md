# Server Access

## Euler (department institutional account)

```bash
ssh -J <user>@ssh2.inf.utfsm.cl <user>@euler.inf.utfsm.cl -i ~/.ssh/id_ed25519_euler
```

Password: `<password>`

## Tarron

```bash
ssh tarron.xionto.cl -p 2222 -l <user>
```

Password: `<password>`

## Handy Commands

Copy checkpoints from Tarron to local:

```bash
scp -P 2222 <user>@tarron.xionto.cl:/home/<user>/top_tagging/checkpoints/bnn/best_model.pt \
    checkpoints/bnn/best_model.pt

scp -P 2222 <user>@tarron.xionto.cl:/home/<user>/top_tagging/checkpoints/resnet50/best_model.pt \
    checkpoints/resnet50/best_model.pt
```

Copy training figures from Tarron to local:

```bash
scp -P 2222 <user>@tarron.xionto.cl:/home/<user>/top_tagging/figures/loss_bnn.png \
    figures/loss_bnn.png

scp -P 2222 <user>@tarron.xionto.cl:/home/<user>/top_tagging/figures/loss_resnet50.png \
    figures/loss_resnet50.png

scp -P 2222 <user>@tarron.xionto.cl:/home/<user>/top_tagging/figures/loss_particle_net.png \
    figures/loss_particle_net.png
```
