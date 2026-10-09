import AppKit
import CoreLocation
import CoreWLAN
import Foundation
import LocalAuthentication
import Security

final class WifiSetup: NSObject, CLLocationManagerDelegate {
    private let manager = CLLocationManager()
    private let output: URL
    private var finished = false

    init(output: URL) {
        self.output = output
        super.init()
        manager.delegate = self
    }

    func start() {
        switch manager.authorizationStatus {
        case .authorizedAlways, .authorizedWhenInUse:
            readNetwork()
        case .notDetermined:
            manager.requestWhenInUseAuthorization()
        default:
            finish(["error": "Allow Kidi Wi-Fi Setup in Location Services to read the current Wi-Fi name."])
        }
    }

    func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
        guard !finished else { return }
        switch manager.authorizationStatus {
        case .authorizedAlways, .authorizedWhenInUse:
            readNetwork()
        case .denied, .restricted:
            finish(["error": "Current Wi-Fi access was denied by macOS."])
        default:
            break
        }
    }

    private func readNetwork() {
        guard let interface = CWWiFiClient.shared().interface(),
              let ssid = interface.ssid(), !ssid.isEmpty else {
            finish(["error": "macOS did not provide a connected Wi-Fi name."])
            return
        }
        finish(["ssid": ssid])
    }

    private func finish(_ result: [String: String]) {
        guard !finished else { return }
        finished = true
        do {
            let data = try JSONSerialization.data(withJSONObject: result)
            try data.write(to: output, options: .atomic)
            try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: output.path)
        } catch {
            fputs("Could not write the private Wi-Fi setup response.\n", stderr)
            exit(1)
        }
        exit(result["error"] == nil ? 0 : 1)
    }
}

func readSelectedCredential() {
    do {
        let input = FileHandle.standardInput.readDataToEndOfFile()
        guard let request = try JSONSerialization.jsonObject(with: input) as? [String: String],
              let ssid = request["ssid"], !ssid.isEmpty else {
            throw NSError(domain: "KidiWifiSetup", code: 1)
        }
        let user = NSUserName()
        let alert = NSAlert()
        alert.messageText = "Allow access to this Wi-Fi credential?"
        alert.informativeText = """
        Network: \(String(reflecting: ssid))
        Mac account: \(user)

        Kidi Wi-Fi Setup will read only this network's saved Wi-Fi password for your USB accessory. It will cache it locally in .kidi.wifi.txt, restricted to your account and ignored by Git, so future setup does not ask again. It will not read other credentials, display the password, or send it to an internet service.
        """
        alert.addButton(withTitle: "Allow Selected Wi-Fi Credential")
        alert.addButton(withTitle: "Cancel")
        NSApp.activate(ignoringOtherApps: true)
        guard alert.runModal() == .alertFirstButtonReturn else {
            FileHandle.standardOutput.write(try JSONSerialization.data(withJSONObject: ["error": "Selected Wi-Fi credential access was cancelled.", "code": "CANCELLED"]))
            exit(1)
        }
        let authorization = LAContext()
        authorization.localizedReason = "Read only the saved Wi-Fi password for \(ssid), authorized by Mac account \(user), for your Kidi USB accessory."
        let query: [CFString: Any] = [
            kSecClass: kSecClassGenericPassword,
            kSecAttrAccount: ssid,
            kSecAttrDescription: "AirPort network password",
            kSecMatchLimit: kSecMatchLimitOne,
            kSecReturnData: true,
            kSecUseAuthenticationContext: authorization
        ]
        var item: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &item)
        guard status == errSecSuccess, let data = item as? Data,
              let password = String(data: data, encoding: .utf8) else {
            let code = status == errSecUserCanceled || status == errSecAuthFailed ? "DENIED" : "UNAVAILABLE"
            FileHandle.standardOutput.write(try JSONSerialization.data(withJSONObject: ["error": "macOS did not authorize access to the selected Wi-Fi credential.", "code": code]))
            exit(1)
        }
        FileHandle.standardOutput.write(try JSONSerialization.data(withJSONObject: ["password": password]))
        exit(0)
    } catch {
        fputs("Could not process the selected Wi-Fi credential request.\n", stderr)
        exit(1)
    }
}
let application = NSApplication.shared
application.setActivationPolicy(.accessory)
if CommandLine.arguments.count == 2 && CommandLine.arguments[1] == "credential" {
    readSelectedCredential()
}
guard CommandLine.arguments.count == 2 else {
    fputs("Expected a private response-file path.\n", stderr)
    exit(1)
}
let setup = WifiSetup(output: URL(fileURLWithPath: CommandLine.arguments[1]))
DispatchQueue.main.async { setup.start() }
application.run()
