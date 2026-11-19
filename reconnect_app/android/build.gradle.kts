allprojects {
    repositories {
        google()
        mavenCentral()
    }
}

val newBuildDir: Directory =
    rootProject.layout.buildDirectory
        .dir("../../build")
        .get()
rootProject.layout.buildDirectory.value(newBuildDir)

subprojects {
    val newSubprojectBuildDir: Directory = newBuildDir.dir(project.name)
    project.layout.buildDirectory.value(newSubprojectBuildDir)
}
subprojects {
    project.evaluationDependsOn(":app")
}

// camera_android_camerax 0.6.30 compiles against CameraX 1.5.3.  With the
// current Android Gradle Plugin, the CameraX transitive dependency below is
// absent from that plugin's Java compile classpath, causing
// `CallbackToFutureAdapter not found`. Declare it explicitly for the plugin.
subprojects {
    if (name == "camera_android_camerax") {
        afterEvaluate {
            dependencies.add(
                "implementation",
                "androidx.concurrent:concurrent-futures:1.1.0",
            )
        }
    }
}

tasks.register<Delete>("clean") {
    delete(rootProject.layout.buildDirectory)
}
