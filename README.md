# best_push_hour
Recommender system for predicting best push time

В ноутбуке разработка-идеи push_best_time_rnd.ipynb

Докер запускается командой (при наличии папки ```data```)

```
docker run --volume "$(pwd)/data:/srv/data" best_push_time:dev
```

Другой вариант - можно было вынести VOLUME в docker-compose

Из-за того, что scikit-surprise использует gcc, пришлось в docker использовать python:3.8 не slim версию
