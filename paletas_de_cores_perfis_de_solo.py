2023-05-19 00:45:39.233 Uncaught app exception

Traceback (most recent call last):

  File "/home/appuser/venv/lib/python3.9/site-packages/streamlit/runtime/scriptrunner/script_runner.py", line 565, in _run_script

    exec(code, module.__dict__)

  File "/app/paletas_de_cores_perfis_de_solo./paletas_de_cores_perfis_de_solo.py", line 110, in <module>

    main()

  File "/app/paletas_de_cores_perfis_de_solo./paletas_de_cores_perfis_de_solo.py", line 105, in main

    result, colors, canvas_image = canvas.generate()

  File "/app/paletas_de_cores_perfis_de_solo./paletas_de_cores_perfis_de_solo.py", line 45, in generate

    quantified_image, colors = self.quantification(clean_img)

AttributeError: 'Canvas' object has no attribute 'quantification'

