# Podcast — rag-pipeline-eval: los tres flujos explicados fácil (~10 min)

Voz: es-AR-TomasNeural · Guión nivel principiante

---

Hola, ¿cómo va? Hoy te voy a contar, en unos diez minutos, cómo funciona uno de los proyectos del portfolio. Se llama rag-pipeline-eval. Sé que el nombre suena a chino técnico, pero te prometo una cosa: lo vas a entender aunque de inteligencia artificial sepas poco y nada. Vamos a arrancar de cero, tranquilos.

Empecemos con una imagen. Pensá en una biblioteca enorme, con miles de documentos: manuales, informes, PDFs, notas. Y pensá en un bibliotecario que se conoce todo eso de memoria y te responde cualquier pregunta que le hagas. La gracia de este proyecto es construir ese bibliotecario, pero hecho con inteligencia artificial. Y con una vuelta de tuerca que casi nadie hace: además de responder, medimos si responde bien. Esa última parte, la de medir, es la más valiosa, y ya vas a ver por qué.

Antes que nada, ¿qué quiere decir RAG? Son tres letras en inglés que significan "generación de respuestas aumentada con búsqueda". En castellano bien simple: la inteligencia artificial no inventa desde la nada. Primero busca en tus documentos, y recién ahí arma la respuesta. Por eso contesta sobre tu información, la tuya, y no sobre cualquier cosa que leyó dando vueltas por internet.

Ahora, lo importante: todo esto tiene tres flujos, tres momentos distintos. El primero es guardar la data. El segundo es consultarla. Y el tercero es evaluar, o sea, controlar si la respuesta estuvo buena. Vamos uno por uno, con calma.

Flujo número uno: cómo se agarra la data y cómo se guarda. Este paso se hace una sola vez, o cada vez que sumás documentos nuevos. Es como ordenar la biblioteca antes de abrirle las puertas al público.

Primero, agarramos los documentos. Pueden ser PDFs, páginas web, textos, lo que tengas a mano. La computadora los abre y los lee.

Segundo, los cortamos en pedacitos. A esto, en la jerga, se le dice "chunking", que es simplemente trocear. ¿Y por qué cortarlos? Porque si le das a la inteligencia artificial un manual de trescientas páginas de una sola vez, se marea, se pierde. En cambio, si lo partís en fragmentos de un par de párrafos, cada pedacito habla de una cosa concreta y después es facilísimo de encontrar. Es como pasar de un libro macizo a un montón de fichas ordenadas.

Y ahora viene la parte que suena más rara: eso de "guardar la data". Porque acá no la guardamos como texto común y corriente. La convertimos en números. Esto se llama "embeddings". Te lo explico con una imagen y lo vas a entender enseguida. Imaginate un mapa gigante, donde cada idea tiene una coordenada, un lugar. Las cosas parecidas quedan cerca en el mapa; las cosas distintas quedan lejos. Por ejemplo, "perro" y "gato" caerían cerquita, porque son animales. Y "perro" y "factura" caerían re lejos, porque no tienen nada que ver. Bueno: cada pedacito de tu documento se transforma en un puntito en ese mapa de significados. A ese puntito, esa lista de números que representa el significado del texto, se le dice "vector".

¿Y dónde se guardan todos esos puntos? En algo que se llama "vector store", que es una base de datos especial, pensada para guardar vectores. En este proyecto usamos una que se llama Chroma, y corre en tu propia máquina, sin mandar nada afuera. Ahí queda todo tu conocimiento convertido en coordenadas, listo para buscar por significado, y no por palabra exacta. Y esa es la verdadera magia: vos podés preguntar con tus propias palabras, y el sistema encuentra lo que quiere decir lo mismo, aunque no hayas usado ni una de las palabras que están en el documento.

Entonces, para resumir el flujo uno: agarramos documentos, los cortamos en fichas, convertimos cada ficha en un punto dentro de un mapa de significados, y guardamos ese mapa. Listo. La biblioteca quedó ordenada y lista para atender.

Vamos al flujo número dos: quién consulta, y cómo. Este flujo pasa cada vez que alguien hace una pregunta. Y ese alguien puede ser un usuario, una aplicación, o vos mismo desde una página web.

Cuando llega la pregunta, por ejemplo, "¿cuánto dura la garantía del producto?", pasa lo siguiente. Primero, esa pregunta también se convierte en un punto en el mapa, con la misma técnica de los embeddings que usamos antes. Segundo, vamos al mapa y buscamos los pedacitos que quedaron más cerca de la pregunta. Esos, seguramente, son los fragmentos de tus documentos que hablan justo de la garantía. A este paso se le dice "recuperar el contexto".

Tercero, agarramos esos fragmentos y se los pasamos al modelo de lenguaje, que es la inteligencia artificial que redacta, junto con la pregunta. Y le decimos algo así como: "Con esta información, y solamente con esta, respondé la pregunta". Entonces la IA arma una respuesta en lenguaje natural, bien redactada, pero apoyándose en tus documentos. Y hasta te puede decir de qué fragmento sacó cada dato. Eso es clave: te da la respuesta y también la fuente, así podés verificar.

Y un detalle lindo de este proyecto: esa inteligencia artificial que redacta puede ser un modelo que corre gratis, en tu propia computadora, o uno más potente en la nube, como Claude. Y lo cambiás con una sola línea de configuración, sin tocar nada del código.

Resumiendo el flujo dos: llega la pregunta, la convertimos en coordenada, buscamos los pedacitos más parecidos, y la inteligencia artificial redacta la respuesta usando esos pedacitos. El bibliotecario te contestó, y encima con las fuentes en la mano.

Y ahora sí, el flujo número tres, el que hace especial a este proyecto: evaluar. O sea, medir si la respuesta fue buena. Porque, ojo con esto, una inteligencia artificial puede sonar segurísima y estar diciendo cualquier verdura. Si no medís, es una caja negra: le crees o no le crees, pero no tenés con qué respaldarte.

¿Y cómo se mide? Preparamos de antemano una lista de preguntas con sus respuestas correctas, como un examen del que ya tenemos las soluciones. Le damos esas preguntas al sistema y comparamos. Pero acá viene lo más interesante: para juzgar, usamos otra inteligencia artificial, como si fuera un profesor que corrige el examen. Le decimos: "mirá esta respuesta y esta fuente, y decime qué tan bien está".

Y se miden cuatro cosas. Te las cuento en criollo. La primera es fidelidad: ¿la respuesta se basa de verdad en los documentos, o la IA se puso a inventar? Si la fidelidad es alta, quiere decir que no alucina. La segunda es relevancia: ¿contestó lo que se le preguntó, o se fue por las ramas? Y la tercera y la cuarta tienen que ver con la búsqueda: ¿trajo los fragmentos correctos y completos, o trajo ruido y se olvidó de cosas importantes?

En la primera medición de este proyecto, la relevancia dio muy bien: las respuestas apuntan justo a la pregunta. Pero la búsqueda dio floja, y eso arrastra la fidelidad para abajo. ¿Y sabés qué? Está perfecto que se vea eso. Porque ese es justamente el punto. Los números te dicen dónde mejorar. No es "anda o no anda", es "acá está el cuello de botella, metele fichas por este lado". Podés cambiar la forma en que cortás los documentos, podés traer más contenido, podés usar un modelo mejor, y después volver a medir para comparar. Antes decidías por intuición; ahora decidís con datos en la mano.

Bueno, vamos a los beneficios, que es lo que más te interesa. Uno: la inteligencia artificial responde sobre tu información, no sobre lo que aprendió de internet, así que sirve para datos privados de tu empresa. Dos: te da las fuentes, así que podés confiar y verificar. Tres: detecta cuándo la IA está alucinando, que es el mayor riesgo de estas herramientas. Cuatro: es reproducible y medible, cada mejora se demuestra con números, no con "a mí me parece". Y cinco: corre gratis en tu máquina si querés, o en la nube si necesitás más calidad, sin reescribir absolutamente nada.

¿Y dónde lo podés aplicar? En un montón de lugares. Un asistente de soporte que responde usando los manuales de tu producto. Un buscador interno para que los empleados encuentren políticas, procedimientos o documentación técnica sin volverse locos. Un bot que contesta preguntas sobre contratos o normativas legales, citando el artículo exacto. Un ayudante para médicos que consulta las guías clínicas. O algo tan simple como un chat sobre la documentación de tu propio proyecto, para que alguien que recién entra al equipo se ponga al día en horas en vez de semanas. En cualquier lado donde tengas muchos documentos y gente que necesita respuestas rápidas y confiables, esto encaja perfecto.

Y así de simple, tres flujos. Guardás la data convirtiéndola en un mapa de significados. Consultás buscando en ese mapa y dejando que la inteligencia artificial redacte usando esas fuentes. Y evaluás, para saber con números si está respondiendo bien. Lo que separa a este proyecto de un demo cualquiera es justamente ese tercer paso: no asume que la IA es buena, lo demuestra. Y eso, en el mundo real, es la diferencia entre un jueguito y algo en lo que de verdad podés confiar. Bueno, hasta acá llegamos por hoy. ¡Nos vemos en el próximo episodio!
